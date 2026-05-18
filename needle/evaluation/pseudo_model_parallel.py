"""
Parallelized pseudo-model implementation with level-wise execution.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from needle.utils.logging import ColorFormatter
from needle.utils.results import AggregationEdge, AggregationMethod, DAGSnapshot

logger = ColorFormatter.get_logger("pseudo_model_parallel")


class PseudoModelParallel(nn.Module):
    """
    Parallelized composite model with level-wise execution.

    Improvements over base PseudoModel:
    - All fold models execute in parallel (independent nodes)
    - Aggregations at same DAG level execute in parallel
    - GPU stream-based parallelization for CUDA devices
    """

    def __init__(self, snapshot_path: str, device: Optional[str] = None, num_workers: int = 4):
        super().__init__()
        self.snapshot = DAGSnapshot.from_json(snapshot_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        logger.info(f"Loading PseudoModelParallel on device: {self.device}")

        self._load_models()
        self._build_execution_levels()

    def _load_models(self):
        """Load all leaf node models (same as base implementation)"""
        # [Same implementation as PseudoModel._load_models()]
        self.models = nn.ModuleDict()
        fold_nodes = [
            (node_id, metadata) for node_id, metadata in self.snapshot.nodes.items() if metadata.task_type == "fold"
        ]
        logger.info(f"Loading {len(fold_nodes)} fold models...")

        for node_id, metadata in fold_nodes:
            try:
                checkpoint = torch.load(metadata.checkpoint_path, map_location=self.device, weights_only=False)

                estimator_name = metadata.estimator_name
                estimators_config = self.snapshot.config_snapshot.get("estimators")
                estimator_config = estimators_config.get(estimator_name)
                model_config = estimator_config.get("model_override")
                dataset_config = estimator_config.get("dataset_override")

                model = hydra.utils.instantiate(
                    OmegaConf.create(model_config),
                    dataset_config=dataset_config,
                )

                if hasattr(model, "configure_model"):
                    model.configure_model()

                state_dict = checkpoint.get("state_dict", checkpoint)
                state_dict = self._clean_state_dict(state_dict, model)
                model.load_state_dict(state_dict)

                if hasattr(model, "model"):
                    model = model.model

                model.eval()
                model.to(self.device)

                for param in model.parameters():
                    param.requires_grad = False

                self.models[node_id] = model

            except Exception as e:
                logger.error(f"Failed to load model {node_id}: {e}")
                raise

        logger.info(f"Successfully loaded {len(self.models)} models")

    def _clean_state_dict(self, state_dict: Dict, model: nn.Module) -> Dict:
        """Clean state dict by handling key prefix mismatches."""
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        if model_keys == checkpoint_keys:
            return state_dict

        common_prefixes = ["model.", "module.", "_orig_mod."]
        for prefix in common_prefixes:
            if all(k.startswith(prefix) for k in checkpoint_keys):
                cleaned = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}
                if set(cleaned.keys()) == model_keys:
                    logger.debug(f"Stripped prefix '{prefix}' from checkpoint keys")
                    return cleaned

        return state_dict

    def _build_execution_levels(self):
        """
        Build execution levels for parallel processing.

        Each level contains edges that can be executed in parallel because
        their source nodes are all available from previous levels.
        """
        self.execution_levels: List[List[AggregationEdge]] = []

        # Track which nodes have been processed
        processed_nodes = set(self.models.keys())  # Fold nodes start as processed
        remaining_edges = list(self.snapshot.edges)

        # Build levels bottom-up
        while remaining_edges:
            current_level = []

            # Find all edges whose sources are ready
            for edge in remaining_edges[:]:
                if all(src in processed_nodes for src in edge.source_nodes):
                    current_level.append(edge)
                    remaining_edges.remove(edge)

            if not current_level:
                # No progress made - cycle detected or missing nodes
                raise ValueError(f"Cannot build execution levels. Remaining edges: {remaining_edges}")

            # Add target nodes from this level to processed AFTER building the level
            # This prevents edges that depend on current level outputs from being
            # added to the same level
            for edge in current_level:
                processed_nodes.add(edge.target_node)

            self.execution_levels.append(current_level)

        logger.info(f"Built {len(self.execution_levels)} execution levels")
        for i, level in enumerate(self.execution_levels):
            logger.debug(f"Level {i}: {len(level)} parallel operations")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallelized forward pass through ensemble hierarchy.

        Strategy:
        1. Execute all fold models in parallel
        2. Process aggregations level-by-level
        3. Within each level, execute aggregations in parallel
        """
        x = x.to(self.device)
        outputs_cache: Dict[str, torch.Tensor] = {}

        # PARALLEL EXECUTION OF FOLD MODELS
        if self.device.startswith("cuda"):
            # GPU: Use CUDA streams for parallel execution
            outputs_cache = self._parallel_folds_gpu(x)
        else:
            # CPU: Use thread pool for parallel execution
            outputs_cache = self._parallel_folds_cpu(x)

        # LEVEL-WISE PARALLEL AGGREGATION
        for level_idx, level_edges in enumerate(self.execution_levels):
            level_results = self._execute_level_parallel(level_edges, outputs_cache)
            outputs_cache.update(level_results)

        # Return root node output
        mean = outputs_cache[self.snapshot.root_node]
        std = outputs_cache.get(f"{self.snapshot.root_node}_std", torch.zeros_like(mean))

        return mean, std

    def _parallel_folds_gpu(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute fold models in parallel using CUDA streams."""
        outputs = {}
        streams = [torch.cuda.Stream() for _ in range(min(len(self.models), 8))]

        for idx, (node_id, model) in enumerate(self.models.items()):
            stream = streams[idx % len(streams)]
            with torch.cuda.stream(stream):
                outputs[node_id] = model(x.clone())

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()

        return outputs

    def _parallel_folds_cpu(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute fold models in parallel using thread pool."""
        outputs = {}

        def run_model(node_id: str, model: nn.Module, x: torch.Tensor):
            with torch.no_grad():
                return node_id, model(x)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(run_model, node_id, model, x.clone()): node_id for node_id, model in self.models.items()
            }

            for future in as_completed(futures):
                node_id, output = future.result()
                outputs[node_id] = output

        return outputs

    def _execute_level_parallel(
        self, level_edges: List[AggregationEdge], outputs_cache: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute all aggregations in a level in parallel."""
        level_results = {}

        if len(level_edges) == 1:
            # Single edge - no parallelization needed
            edge = level_edges[0]
            source_outputs = [outputs_cache[node] for node in edge.source_nodes]
            aggregated, variance = self._aggregate(source_outputs, edge)
            level_results[edge.target_node] = aggregated
            level_results[f"{edge.target_node}_std"] = variance
        else:
            # Multiple edges - parallelize aggregations
            def aggregate_edge(edge: AggregationEdge):
                source_outputs = [outputs_cache[node] for node in edge.source_nodes]
                aggregated, variance = self._aggregate(source_outputs, edge)
                return edge.target_node, aggregated, variance

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(aggregate_edge, edge) for edge in level_edges]

                for future in as_completed(futures):
                    target_node, aggregated, variance = future.result()
                    level_results[target_node] = aggregated
                    level_results[f"{target_node}_std"] = variance

        return level_results

    def _aggregate(self, outputs: List[torch.Tensor], edge: AggregationEdge) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate outputs according to edge method."""
        if edge.method == AggregationMethod.MEAN:
            stacked = torch.stack(outputs, dim=0)
            aggregated = stacked.mean(dim=0)
            std = stacked.std(dim=0)

        elif edge.method == AggregationMethod.SUM:
            stacked = torch.stack(outputs, dim=0)
            aggregated = stacked.sum(dim=0)
            variances = stacked.var(dim=0)
            std = torch.sqrt(variances.sum(dim=0, keepdim=True).expand_as(variances))

        elif edge.method == AggregationMethod.BEST:
            best_idx = self._select_best_model(edge.source_nodes, edge.metric_key)
            aggregated = outputs[best_idx]
            std = torch.zeros_like(aggregated)

        elif edge.method == AggregationMethod.WEIGHTED_MEAN:
            if edge.weights is None:
                raise ValueError("Weights required for weighted_mean aggregation")

            weights = torch.tensor(edge.weights, device=self.device, dtype=torch.float32)
            weights = weights / weights.sum()

            aggregated = sum(w * out for w, out in zip(weights, outputs))
            stacked = torch.stack(outputs, dim=0)
            weighted_var = (weights.view(-1, *([1] * (stacked.dim() - 1))) * (stacked - aggregated) ** 2).sum(dim=0)
            std = torch.sqrt(weighted_var)

        else:
            raise ValueError(f"Unknown aggregation method: {edge.method}")

        return aggregated, std

    def _select_best_model(self, node_ids: List[str], metric_key: Optional[str]) -> int:
        """Select best model based on validation metric"""
        if metric_key is None:
            raise ValueError("metric_key required for 'best' aggregation")

        metrics = [self.snapshot.nodes[nid].metrics.get(metric_key, float("inf")) for nid in node_ids]
        return int(torch.tensor(metrics).argmin())


class NEEDLEParallel:
    """High-level API for parallelized NEEDLE model evaluation"""

    def __init__(self, snapshot_path: str, device: Optional[str] = None, num_workers: int = 4):
        self.model = PseudoModelParallel(snapshot_path, device, num_workers)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Simple evaluation returning mean prediction"""
        self.model.eval()
        with torch.no_grad():
            mean, _ = self.model(x)
        return mean

    def eval_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation with uncertainty quantification"""
        self.model.eval()
        with torch.no_grad():
            mean, std = self.model(x)
        return mean, std
