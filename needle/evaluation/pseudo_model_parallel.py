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

    def __init__(
        self,
        snapshot_path: str,
        device: Optional[str] = None,
        num_workers: int = 4,
        n_gpus: int = 1,
        n_streams_per_device: int = 0,
    ):
        """
        Args:
            n_gpus: Number of physical GPUs to distribute fold models across (round-robin).
                Ignored (falls back to a single device) when CUDA is unavailable. If more GPUs
                are requested than are available, the available count is used and a warning is
                logged.
            n_streams_per_device: Number of CUDA streams to create per physical GPU. Each fold is
                round-robined across these streams, letting the hardware scheduler overlap kernels
                when SM occupancy allows. 0 (default) means one stream per fold on that device
                (maximum potential concurrency). Use 1 to force serial execution within a device.
        """
        super().__init__()
        self.snapshot = DAGSnapshot.from_json(snapshot_path)
        self.num_workers = num_workers
        self._n_streams_per_device_cfg = n_streams_per_device  # 0 = auto

        # Build the list of devices to round-robin folds across. If n_gpus > 1 and CUDA is
        # available, use cuda:0 ... cuda:n_gpus-1. Falls back gracefully to a single device when
        # fewer GPUs are present, or to CPU when CUDA is unavailable.
        if device is None and torch.cuda.is_available():
            n_available = torch.cuda.device_count()
            n_use = min(n_gpus, n_available)
            if n_use < n_gpus:
                logger.warning(f"Requested {n_gpus} GPU(s) but only {n_available} available. Using {n_use}.")
            self.devices = [f"cuda:{i}" for i in range(n_use)]
        elif device is not None:
            self.devices = [device]
        else:
            self.devices = ["cpu"]

        # Primary device used for aggregation outputs and non-fold tensors.
        self.device = self.devices[0]
        logger.info(f"Loading PseudoModelParallel across {len(self.devices)} device(s): {self.devices}")

        self._load_models()
        self._build_execution_levels()

        # Resolve n_streams_per_device now that we know how many folds there are.
        # 0 = one stream per fold on each device (maximum scheduled concurrency).
        n_folds_per_dev = (
            max(sum(1 for d in self.fold_devices.values() if d == dev) for dev in self.devices)
            if self.fold_devices
            else 1
        )
        if self._n_streams_per_device_cfg == 0:
            self.n_streams_per_device = max(1, n_folds_per_dev)
        else:
            self.n_streams_per_device = max(1, self._n_streams_per_device_cfg)
        logger.info(f"Using {self.n_streams_per_device} CUDA stream(s) per device ({n_folds_per_dev} fold(s) per device)")

    def _load_models(self):
        """Load fold models, distributing them across self.devices round-robin."""
        self.models = nn.ModuleDict()
        # Map each node_id -> the device it lives on.
        self.fold_devices: Dict[str, str] = {}
        # Map each node_id -> stream index (within its device) to use.
        self.fold_stream_idx: Dict[str, int] = {}
        fold_nodes = [
            (node_id, metadata) for node_id, metadata in self.snapshot.nodes.items() if metadata.task_type == "fold"
        ]
        logger.info(f"Loading {len(fold_nodes)} fold models across {self.devices}...")

        for idx, (node_id, metadata) in enumerate(fold_nodes):
            target_device = self.devices[idx % len(self.devices)]
            # Which stream slot within that device this fold will use. Assignment is finalized
            # post-init once n_streams_per_device is known; store the raw per-device fold index
            # for now and resolve it in _parallel_folds_gpu using self.n_streams_per_device.
            self.fold_stream_idx[node_id] = idx // len(self.devices)
            try:
                checkpoint = torch.load(metadata.checkpoint_path, map_location=target_device, weights_only=False)

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
                model.to(target_device)
                self.fold_devices[node_id] = target_device

                for param in model.parameters():
                    param.requires_grad = False

                self.models[node_id] = model

            except Exception as e:
                logger.error(f"Failed to load model {node_id} on {target_device}: {e}")
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
        """Execute fold models in parallel using multiple CUDA streams per device.

        Each physical device gets ``self.n_streams_per_device`` independent CUDA streams. Fold
        models are round-robined across those streams so the GPU hardware scheduler can overlap
        kernel execution whenever SM occupancy permits (especially beneficial for small models
        that leave most SMs idle during a single forward pass).

        With ``n_streams_per_device == n_folds_per_device`` every fold gets its own stream --
        maximum scheduling latitude. With ``== 1`` all folds on a device share one stream and
        execute serially.
        """
        outputs: Dict[str, torch.Tensor] = {}

        # (device, stream_index) -> Stream
        stream_map: Dict[tuple, torch.cuda.Stream] = {
            (dev, s): torch.cuda.Stream(device=dev) for dev in self.devices for s in range(self.n_streams_per_device)
        }

        # Pre-scatter the input to every device exactly once using stream 0 on each device. This
        # avoids issuing N_folds redundant PCIe copies of the same tensor when len(self.devices) > 1.
        x_per_dev: Dict[str, torch.Tensor] = {}
        for dev in self.devices:
            if dev == self.device:
                x_per_dev[dev] = x  # already on primary device - no copy
            else:
                with torch.cuda.stream(stream_map[(dev, 0)]):
                    x_per_dev[dev] = x.to(dev, non_blocking=True)

        for node_id, model in self.models.items():
            dev = self.fold_devices[node_id]
            s_idx = self.fold_stream_idx[node_id] % self.n_streams_per_device
            stream = stream_map[(dev, s_idx)]
            with torch.cuda.stream(stream):
                outputs[node_id] = model(x_per_dev[dev])

        # Synchronise every stream on every device.
        for (dev, _), stream in stream_map.items():
            with torch.cuda.device(dev):
                stream.synchronize()

        # Move results back to primary device for aggregation. Issue all transfers non-blocking
        # first, then synchronise once per secondary device so the PCIe copies overlap as much as
        # possible.
        if len(self.devices) > 1:
            gather_streams = {
                dev: torch.cuda.Stream(device=self.device) for dev in self.devices if dev != self.device
            }
            gathered: Dict[str, torch.Tensor] = {}
            for k, v in outputs.items():
                if v.device.type == "cuda" and str(v.device) != self.device:
                    with torch.cuda.stream(gather_streams[str(v.device)]):
                        gathered[k] = v.to(self.device, non_blocking=True)
                else:
                    gathered[k] = v
            for gs in gather_streams.values():
                gs.synchronize()
            outputs = gathered

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

    def __init__(
        self,
        snapshot_path: str,
        device: Optional[str] = None,
        num_workers: int = 4,
        n_gpus: int = 1,
        n_streams_per_device: int = 0,
    ):
        self.model = PseudoModelParallel(snapshot_path, device, num_workers, n_gpus, n_streams_per_device)

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
