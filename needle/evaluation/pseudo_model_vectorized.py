"""
Vectorized pseudo-model using torch.vmap for maximum parallelization.

This approach is optimal when all models have identical architectures.
"""

from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from functorch import vmap
from omegaconf import OmegaConf

from needle.utils.logging import ColorFormatter
from needle.utils.results import AggregationEdge, AggregationMethod, DAGSnapshot

logger = ColorFormatter.get_logger("pseudo_model_vectorized")


class VectorizedEnsemble(nn.Module):
    """
    Wrapper that vectorizes multiple models with identical architecture.

    Instead of:
        for model in models: output = model(x)

    We do:
        outputs = vectorized_models(x)  # Single batched operation
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.num_models = len(models)

        # Stack all model parameters into batched tensors
        self.batched_params = self._stack_parameters(models)

        # Use first model as template for forward pass
        self.template_model = models[0]

    def _stack_parameters(self, models: List[nn.Module]) -> Dict[str, torch.Tensor]:
        """Stack parameters from multiple models into batched tensors."""
        batched = {}

        # Get parameter names from first model
        param_names = [name for name, _ in models[0].named_parameters()]

        for name in param_names:
            # Stack parameters across models (adds batch dimension)
            params = [dict(model.named_parameters())[name] for model in models]
            batched[name] = torch.stack(params, dim=0)

        return batched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward pass across all models.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Output tensor [num_models, batch_size, ...]
        """

        # Define stateless forward function
        def model_fn(params, x):
            # Temporarily set parameters
            return torch.func.functional_call(self.template_model, params, (x,))

        # Vectorize over model dimension
        batched_fn = vmap(model_fn, in_dims=(0, None))

        # Execute all models in parallel
        outputs = batched_fn(self.batched_params, x)

        return outputs


class PseudoModelVectorized(nn.Module):
    """
    Fully vectorized pseudo-model for maximum GPU utilization.

    Key optimizations:
    1. Batched model execution using torch.vmap
    2. Vectorized aggregations (no loops)
    3. Fused operations where possible
    """

    def __init__(self, snapshot_path: str, device: Optional[str] = None):
        super().__init__()
        self.snapshot = DAGSnapshot.from_json(snapshot_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading PseudoModelVectorized on device: {self.device}")

        self._load_models()
        self._build_vectorized_graph()

    def _load_models(self):
        """Load models and group by architecture."""
        self.models = nn.ModuleDict()
        self.model_groups: Dict[str, List[str]] = {}  # architecture -> node_ids

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

                # Group by architecture (estimator_name as proxy)
                if estimator_name not in self.model_groups:
                    self.model_groups[estimator_name] = []
                self.model_groups[estimator_name].append(node_id)

            except Exception as e:
                logger.error(f"Failed to load model {node_id}: {e}")
                raise

        logger.info(f"Successfully loaded {len(self.models)} models")
        logger.info(f"Grouped into {len(self.model_groups)} architectures")

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
                    return cleaned

        return state_dict

    def _build_vectorized_graph(self):
        """Build vectorized execution plan."""
        # Create vectorized ensembles for each architecture group
        self.vectorized_ensembles: Dict[str, VectorizedEnsemble] = {}

        for arch_name, node_ids in self.model_groups.items():
            models = [self.models[nid] for nid in node_ids]

            # Check if all models have identical architecture
            if self._check_identical_architecture(models):
                self.vectorized_ensembles[arch_name] = VectorizedEnsemble(models)
                logger.info(f"Created vectorized ensemble for {arch_name} with {len(models)} models")
            else:
                logger.warning(f"Models in {arch_name} have different architectures, falling back to sequential")

        # Build execution levels (same as parallel version)
        self.execution_levels: List[List[AggregationEdge]] = []
        processed_nodes = set(self.models.keys())
        remaining_edges = list(self.snapshot.edges)

        while remaining_edges:
            current_level = []
            for edge in remaining_edges[:]:
                if all(src in processed_nodes for src in edge.source_nodes):
                    current_level.append(edge)
                    remaining_edges.remove(edge)
                    processed_nodes.add(edge.target_node)

            if not current_level:
                raise ValueError(f"Cannot build execution levels")

            self.execution_levels.append(current_level)

    def _check_identical_architecture(self, models: List[nn.Module]) -> bool:
        """Check if all models have identical parameter shapes."""
        if len(models) <= 1:
            return True

        ref_shapes = [p.shape for p in models[0].parameters()]

        for model in models[1:]:
            shapes = [p.shape for p in model.parameters()]
            if shapes != ref_shapes:
                return False

        return True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized forward pass.

        All models within an architecture group execute as a single
        batched operation on the GPU.
        """
        x = x.to(self.device)
        outputs_cache: Dict[str, torch.Tensor] = {}

        # VECTORIZED EXECUTION BY ARCHITECTURE GROUP
        for arch_name, vectorized_ensemble in self.vectorized_ensembles.items():
            node_ids = self.model_groups[arch_name]

            # Single batched forward pass for entire group
            batched_outputs = vectorized_ensemble(x)  # [num_models, batch_size, ...]

            # Unpack into cache
            for i, node_id in enumerate(node_ids):
                outputs_cache[node_id] = batched_outputs[i]

        # Fallback for non-vectorized models
        for node_id, model in self.models.items():
            if node_id not in outputs_cache:
                outputs_cache[node_id] = model(x)

        # VECTORIZED AGGREGATION
        for level_edges in self.execution_levels:
            for edge in level_edges:
                source_outputs = [outputs_cache[node] for node in edge.source_nodes]
                aggregated, variance = self._aggregate_vectorized(source_outputs, edge)

                outputs_cache[edge.target_node] = aggregated
                outputs_cache[f"{edge.target_node}_std"] = variance

        mean = outputs_cache[self.snapshot.root_node]
        std = outputs_cache.get(f"{self.snapshot.root_node}_std", torch.zeros_like(mean))

        return mean, std

    def _aggregate_vectorized(
        self, outputs: List[torch.Tensor], edge: AggregationEdge
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fully vectorized aggregation (no loops)."""
        # Stack tensors once
        stacked = torch.stack(outputs, dim=0)  # [num_models, batch, ...]

        if edge.method == AggregationMethod.MEAN:
            aggregated = stacked.mean(dim=0)
            std = stacked.std(dim=0)

        elif edge.method == AggregationMethod.SUM:
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

            # Broadcasting-based weighted mean (no loops)
            weights = torch.tensor(edge.weights, device=self.device, dtype=torch.float32)
            weights = weights / weights.sum()

            # Reshape weights for broadcasting: [num_models, 1, 1, ...]
            weight_shape = [len(weights)] + [1] * (stacked.dim() - 1)
            weights_expanded = weights.view(*weight_shape)

            # Weighted mean via broadcasting
            aggregated = (stacked * weights_expanded).sum(dim=0)

            # Weighted variance
            weighted_var = (weights_expanded * (stacked - aggregated) ** 2).sum(dim=0)
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


class NEEDLEVectorized:
    """High-level API for vectorized NEEDLE model evaluation"""

    def __init__(self, snapshot_path: str, device: Optional[str] = None):
        self.model = PseudoModelVectorized(snapshot_path, device)

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
