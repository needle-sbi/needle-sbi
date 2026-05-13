"""
PyTorch pseudo-model that aggregates multiple trained models.
"""

from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from needle.utils.logging import ColorFormatter
from needle.utils.results import AggregationEdge, AggregationMethod, DAGSnapshot

logger = ColorFormatter.get_logger("pseudo_model")


class PseudoModel(nn.Module):
    """
    Composite model that aggregates multiple trained models according to DAG snapshot.
    Supports multi-level hierarchical aggregation.
    """

    def __init__(self, snapshot_path: str, device: Optional[str] = None):
        super().__init__()
        self.snapshot = DAGSnapshot.from_json(snapshot_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading PseudoModel on device: {self.device}")

        self._load_models()
        self._build_aggregation_graph()

    def _load_models(self):
        """Load all leaf node models (fold-level checkpoints)"""
        self.models = nn.ModuleDict()

        # Count fold nodes
        fold_nodes = [
            (node_id, metadata) for node_id, metadata in self.snapshot.nodes.items() if metadata.task_type == "fold"
        ]
        logger.info(f"Loading {len(fold_nodes)} fold models...")

        for node_id, metadata in fold_nodes:
            try:
                # Try direct checkpoint loading
                logger.debug(f"Loading checkpoint: {metadata.checkpoint_path}")
                checkpoint = torch.load(metadata.checkpoint_path, map_location=self.device, weights_only=False)

                # Extract estimator name from node_id
                # node_id format: "estimator_{name}_systematic_{sys}_ensemble_{ens}_fold_{fold}"
                estimator_name = metadata.estimator_name
                logger.debug(f"Loading model for estimator: {estimator_name}")

                # Get model config from snapshot for this estimator
                estimators_config = self.snapshot.config_snapshot.get("estimators")
                if estimators_config is None:
                    raise ValueError("Estimators configuration not found in snapshot")

                estimator_config = estimators_config.get(estimator_name)
                if estimator_config is None:
                    raise ValueError(f"Estimator {estimator_name} not found in config")

                # Get the model_override config (contains _target_)
                model_config = estimator_config.get("model_override")
                if model_config is None:
                    raise ValueError(
                        f"model_override not found for estimator {estimator_name}. "
                        f"Available keys: {list(estimator_config.keys())}"
                    )

                # Get dataset config (needed for model instantiation)
                dataset_config = estimator_config.get("dataset_override")

                # Instantiate model using Hydra
                logger.debug(f"Instantiating model with _target_: {model_config.get('_target_')}")
                model = hydra.utils.instantiate(
                    OmegaConf.create(model_config),
                    dataset_config=dataset_config,
                )

                # Lightning modules need configure_model() called
                if hasattr(model, "configure_model"):
                    model.configure_model()

                # Load state dict with prefix handling
                state_dict = checkpoint.get("state_dict", checkpoint)
                state_dict = self._clean_state_dict(state_dict, model)
                model.load_state_dict(state_dict)

                # For Lightning modules, extract the actual model for inference
                # to avoid forward() signature mismatches (e.g., requiring target)
                if hasattr(model, "model"):
                    model = model.model

                model.eval()
                model.to(self.device)

                # Freeze all parameters
                for param in model.parameters():
                    param.requires_grad = False

                self.models[node_id] = model
                logger.debug(f"Loaded model: {node_id}")

            except Exception as e:
                logger.error(f"Failed to load model {node_id}: {e}")
                raise

        logger.info(f"Successfully loaded {len(self.models)} models")

    def _clean_state_dict(self, state_dict: Dict, model: nn.Module) -> Dict:
        """Clean state dict by handling key prefix mismatches.

        PyTorch Lightning sometimes saves models with prefixes like 'model.'
        This method strips or adds prefixes to match the model's expected keys.
        """
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        # If keys match exactly, return as-is
        if model_keys == checkpoint_keys:
            return state_dict

        # Try stripping common prefixes from checkpoint
        common_prefixes = ["model.", "module.", "_orig_mod."]
        for prefix in common_prefixes:
            if all(k.startswith(prefix) for k in checkpoint_keys):
                cleaned = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}
                if set(cleaned.keys()) == model_keys:
                    logger.debug(f"Stripped prefix '{prefix}' from checkpoint keys")
                    return cleaned

        # Try adding prefixes to model keys
        for prefix in common_prefixes:
            if all(f"{prefix}{k}" in checkpoint_keys for k in model_keys):
                # Checkpoint has extra prefix, strip it
                cleaned = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}
                logger.debug(f"Stripped prefix '{prefix}' from checkpoint keys")
                return cleaned

        # If we can't resolve automatically, return original and let PyTorch error
        logger.warning(
            f"Could not automatically resolve key mismatch. "
            f"Model expects {len(model_keys)} keys, checkpoint has {len(checkpoint_keys)} keys."
        )
        return state_dict

    def _build_aggregation_graph(self):
        """Build execution graph from edges - topologically sorted"""
        self.edge_map: Dict[str, AggregationEdge] = {}
        self.execution_order: List[AggregationEdge] = []

        # Build edge map
        for edge in self.snapshot.edges:
            self.edge_map[edge.target_node] = edge

        # Topological sort: process edges from leaves to root
        # Start with edges whose sources are all fold nodes
        processed_nodes = set(self.models.keys())  # Fold nodes are already processed

        while len(self.execution_order) < len(self.snapshot.edges):
            for edge in self.snapshot.edges:
                if edge in self.execution_order:
                    continue

                # Check if all source nodes are processed
                if all(src in processed_nodes for src in edge.source_nodes):
                    self.execution_order.append(edge)
                    processed_nodes.add(edge.target_node)
                    break
            else:
                # No edge was added - cycle detected or missing nodes
                remaining = [e for e in self.snapshot.edges if e not in self.execution_order]
                raise ValueError(f"Cannot topologically sort edges. Remaining: {remaining}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble hierarchy.

        Args:
            x: Input tensor

        Returns:
            mean: Aggregated prediction from root node
            std: Standard deviation across models (uncertainty estimate)
        """
        # Ensure input is on correct device
        x = x.to(self.device)

        # Execute all leaf models (fold nodes)
        outputs_cache: Dict[str, torch.Tensor] = {}

        for node_id, model in self.models.items():
            outputs_cache[node_id] = model(x)

        # Aggregate in topological order
        for edge in self.execution_order:
            source_outputs = [outputs_cache[node] for node in edge.source_nodes]
            aggregated, variance = self._aggregate(source_outputs, edge)

            outputs_cache[edge.target_node] = aggregated
            outputs_cache[f"{edge.target_node}_std"] = variance

        # Return root node output
        mean = outputs_cache[self.snapshot.root_node]
        std = outputs_cache.get(f"{self.snapshot.root_node}_std", torch.zeros_like(mean))

        return mean, std

    def _aggregate(self, outputs: List[torch.Tensor], edge: AggregationEdge) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate outputs according to edge method.

        Returns:
            aggregated: Aggregated tensor
            std: Standard deviation (uncertainty)
        """
        if edge.method == AggregationMethod.MEAN:
            stacked = torch.stack(outputs, dim=0)
            aggregated = stacked.mean(dim=0)
            std = stacked.std(dim=0)

        elif edge.method == AggregationMethod.SUM:
            stacked = torch.stack(outputs, dim=0)
            aggregated = stacked.sum(dim=0)
            # For sum, uncertainty is sqrt of sum of variances (assuming independence)
            variances = stacked.var(dim=0)
            std = torch.sqrt(variances.sum(dim=0, keepdim=True).expand_as(variances))

        elif edge.method == AggregationMethod.BEST:
            # Select based on validation metric
            best_idx = self._select_best_model(edge.source_nodes, edge.metric_key)
            aggregated = outputs[best_idx]
            std = torch.zeros_like(aggregated)

        elif edge.method == AggregationMethod.WEIGHTED_MEAN:
            if edge.weights is None:
                raise ValueError("Weights required for weighted_mean aggregation")

            weights = torch.tensor(edge.weights, device=self.device, dtype=torch.float32)
            weights = weights / weights.sum()  # Normalize

            aggregated = sum(w * out for w, out in zip(weights, outputs))

            # Weighted variance
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


class NEEDLE:
    """High-level API for NEEDLE model evaluation"""

    def __init__(self, snapshot_path: str, device: Optional[str] = None):
        self.model = PseudoModel(snapshot_path, device)

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
