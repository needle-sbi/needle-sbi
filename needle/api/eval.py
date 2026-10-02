from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union, get_args
from urllib.parse import parse_qsl

import lightning as L
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from needle.api.aggregation import aggregate_siblings
from needle.api.config import load_config
from needle.utils.config_schema import AggregationSpec, EstimatorConfig, MainConfig
from needle.utils.config_utils import hydra_instantiate, merge_systematic_config
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("eval")

# nested[systematic][ensemble][fold] = ckpt_path
_EstimatorSnapshot = Dict[str, Dict[int, Dict[int, str]]]


def load_snapshot(
    results_path: Path,
    estimator: str,
    snapshot_file: str = "dag_snapshot.json",
) -> _EstimatorSnapshot:
    """Load a snapshot and unflatten the checkpoints of the given estimator.

    Each key is a urlencoded query string ``est=<estimator>&syst=<systematic>&ensem=<ensemble>
    &fold=<fold_index>`` (see `needle.tasks.base.estimator.BaseEstimatorTask.input_model_paths`
    and `needle.tasks.base.main.BaseMainTask.snapshot_as_dict`).

    Args:
        results_path: Directory containing the snapshot file.
        estimator: Name of the estimator whose checkpoints should be loaded.
        snapshot_file: Name of the json snapshot file within ``results_path``. Defaults to  `dag_snapshot.json`

    Returns:
        A nested mapping.

    For example::
    ```
        {
            "nominal": {
                0: {0: "checkpoints/nominal_0_0.ckpt", 1: "checkpoints/nominal_0_1.ckpt"},
                1: {0: "checkpoints/nominal_1_0.ckpt"},
            },
            "shifted": {
                0: {0: "checkpoints/shifted_0_0.ckpt"},
            },
        }
    ```

    The keys are `systematic` names, `ensemble` indices, and `fold` indices,
    respectively. The innermost values are checkpoint paths.

    In the example above, you access the checkpoint for systematics "nominal", ensemble 0 and fold 1 using:

    >>> snapshot = load_snapshot(Path("runs/my_run"), "model_A")
    >>> snapshot["nominal"][0][1]
    "checkpoints/nominal_0_1.ckpt"
    """
    snapshot_path = results_path / snapshot_file

    if not snapshot_path.exists():
        raise FileNotFoundError(f"No dag_snapshot.json found at {snapshot_path}. Run a MainTask to completion first.")

    with open(snapshot_path) as f:
        flat: Dict[str, str] = json.load(f)

    nested: _EstimatorSnapshot = defaultdict(lambda: defaultdict(dict))
    found = False

    for key, ckpt_path in flat.items():
        fields = dict(parse_qsl(key))

        if fields.get("est") != estimator:
            continue

        found = True
        nested[fields["syst"]][int(fields["ensem"])][int(fields["fold"])] = ckpt_path

    if not found:
        raise KeyError(f"No checkpoints for estimator {estimator!r} found in {snapshot_path}")

    return nested


def _clean_state_dict(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """Strip/add common key prefixes (``model.``, ``module.``, ``_orig_mod.``) so a checkpoint's
    state dict matches `model`, handling the wrapping Lightning does around the raw `nn.Module`.
    """
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    if model_keys == checkpoint_keys:
        return state_dict

    common_prefixes = ["model.", "module.", "_orig_mod."]

    for prefix in common_prefixes:
        if all(k.startswith(prefix) for k in checkpoint_keys):
            cleaned = {k[len(prefix) :]: v for k, v in state_dict.items()}
            if set(cleaned.keys()) == model_keys:
                return cleaned

    for prefix in common_prefixes:
        if all(f"{prefix}{k}" in checkpoint_keys for k in model_keys):
            # Keep only the wrapped model's own weights; extras (e.g. a loss module's) carry no prefix.
            return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    logger.warning(
        f"Could not automatically resolve checkpoint/model key mismatch. "
        f"Model expects {len(model_keys)} keys, checkpoint has {len(checkpoint_keys)} keys."
    )
    return state_dict


def _checkpoint_scores(checkpoint: Dict[str, Any]) -> Dict[str, float]:
    """Best monitored score per metric name (e.g. ``{"val_loss": 0.12}``) recorded by Lightning's
    `ModelCheckpoint` callbacks in `checkpoint`; this is what the "best" aggregation compares.
    """
    scores: Dict[str, float] = {}
    for state in checkpoint.get("callbacks", {}).values():
        if isinstance(state, dict) and state.get("monitor") and state.get("best_model_score") is not None:
            scores[state["monitor"]] = float(state["best_model_score"])

    return scores


def _module_device(module: nn.Module) -> torch.device:
    """The device `module` currently lives on, so it stays correct after `.to()`/`.cpu()`/`.cuda()`."""
    tensor = next(chain(module.parameters(), module.buffers()), None)
    return tensor.device if tensor is not None else torch.device("cpu")


ExecutionMode = Literal["sequential", "parallel", "vectorized"]
_EXECUTION_MODES = get_args(ExecutionMode)


def _check_execution(execution: str) -> None:
    if execution not in _EXECUTION_MODES:
        raise ValueError(f"Unknown execution mode: {execution!r}. Expected one of {_EXECUTION_MODES}.")


def _resolve_devices(device: Optional[str], n_gpus: int) -> List[torch.device]:
    """Resolve the primary device (first entry) and, for ``n_gpus > 1``, further devices of the same type.

    Works for any accelerator PyTorch exposes through `torch.accelerator` (CUDA, ROCm, XPU, MPS, ...);
    nothing here is CUDA specific. Without an explicit ``device`` the current accelerator is used,
    falling back to the CPU.

    Does not actually check whether the primary device exists.
    """
    if device is None:
        accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
        primary = torch.device(accelerator) if accelerator is not None else torch.device("cpu")
    else:
        primary = torch.device(device)

    if primary.type == "cpu":
        return [primary]

    backend = getattr(torch, primary.type)
    if primary.index is None:
        primary = torch.device(primary.type, backend.current_device())

    n_available = backend.device_count() if hasattr(backend, "device_count") else 1
    n_use = min(max(n_gpus, 1), n_available)
    if n_use < n_gpus:
        logger.warning(f"Requested {n_gpus} {primary.type} device(s) but only {n_available} available. Using {n_use}.")

    others = [torch.device(primary.type, i) for i in range(n_available) if i != primary.index]
    return [primary, *others[: n_use - 1]]


class _VectorizedSiblings(nn.Module):
    """Batches a group of structurally-identical sibling models into a single `torch.vmap`'d
    forward pass, used by `Estimator(execution="vectorized")`.

    Only `nn.Module.parameters()` are batched. Buffers (e.g. BatchNorm running stats) are taken from
    the first sibling and shared across the batch, so siblings whose buffers differ are rejected with
    a `ValueError` instead of silently producing wrong results.

    The stacked parameters are registered as (non-persistent) buffers, so `.to()` and friends move
    them with the module. They are a copy: the sibling models' own weights stay alive in `Estimator.models`.
    """

    def __init__(self, keys: List[str], models: List[nn.Module], device: torch.device) -> None:
        super().__init__()
        self.keys = keys
        # A tuple, so `models[0]` is not also registered as a child module of this group.
        self._template = (models[0],)

        template_buffers = dict(models[0].named_buffers())
        for key, model in zip(keys, models):
            buffers = dict(model.named_buffers())
            if buffers.keys() != template_buffers.keys() or any(
                not torch.equal(buffers[name].to(device), template_buffers[name].to(device)) for name in buffers
            ):
                raise ValueError(
                    f"Cannot vectorize siblings with differing buffers (model {key!r} differs from {keys[0]!r}); "
                    f"use execution='sequential' or 'parallel' instead."
                )

        params = [dict(model.named_parameters()) for model in models]
        self._param_names = [name for name, _ in models[0].named_parameters()]
        for i, name in enumerate(self._param_names):
            stacked = torch.stack([p[name].detach().to(device) for p in params], dim=0)
            self.register_buffer(self._buffer_name(i), stacked, persistent=False)

    @staticmethod
    def _buffer_name(index: int) -> str:
        # Parameter names contain dots, which `register_buffer` rejects; index by position instead.
        return f"_stacked_{index}"

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        params = {name: getattr(self, self._buffer_name(i)) for i, name in enumerate(self._param_names)}

        def model_fn(params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            return functional_call(self._template[0], params, (x,))

        batched = vmap(model_fn, in_dims=(0, None))(params, x)
        return {key: batched[i] for i, key in enumerate(self.keys)}


class _Leaf(NamedTuple):
    """One trained model: a (systematic, ensemble, fold) combination in the snapshot."""

    systematic: str
    ensemble: int
    fold: int

    @property
    def key(self) -> str:
        return f"syst={self.systematic}&ensem={self.ensemble}&fold={self.fold}"


class _Node(NamedTuple):
    """A prediction at any level of the aggregation tree, with the scores "best" can compare.

    `scores` maps a monitored metric name to the lowest value among the leaves below this node.
    """

    output: torch.Tensor
    scores: Dict[str, float]


class Estimator(nn.Module):
    """Combined inference model for one trained estimator.

    Loads every checkpoint and aggregates them bottom-up into a single ``(mean, std)`` prediction in
    the following way:

        folds                               (via ``expands.folds.aggregation``)
          └─> ensemble members              (via ``expands.ensembles.aggregation``)
              └─> systematic variations     (via ``estimator.systematic_aggregation``)
                  └─> Estimator

    (So really the inverse of the training DAG).

    Each level's `AggregationSpec.method` is one of the built-ins ("mean" / "sum" / "best") or a
    dotted path to a custom callable (see `needle.aggregate_siblings`). "best" picks the sibling with
    the lowest `AggregationSpec.metric_key` (a metric monitored by a `ModelCheckpoint` during training,
    e.g. "val_loss"); a group of siblings is scored by the lowest value among its models.

    The leaf models (one per systematic/ensemble/fold combination) can be evaluated in one of
    three ``execution`` modes, settable at construction and overridable per-call via
    `forward(..., execution=...)`:

        - "sequential" (default): one Python-level forward call per leaf model. Simplest, safe and
            no extra memory overhead.

        - "parallel": every leaf model (resulting from TrainingTask) is submitted to a
            `ThreadPoolExecutor` with ``num_workers`` threads.
            When constructed with ``n_gpus > 1``, runs the models of different devices concurrently.
            Models on the same device still share that device's stream. On CPU the gains are mainly
            for many small models or several devices. Works on any accelerator in principle, not only CUDA.

        - "vectorized": siblings within one systematic are guaranteed to have the identical
            architecture, so they are batched into a single `torch.vmap` call.
            Fastest on GPU for many small sibling models. See `_VectorizedSiblings` for the buffer restrictions.

    Examples:
        >>> from needle.api.eval import Estimator
        >>> model = Estimator("runs/my_run", "model_A")
        >>> mean, std = model(x)
        >>> model_parallel = Estimator("runs/my_run", "model_A", execution="parallel", n_gpus=2)
        >>> mean, std = model_parallel(x)

    If you instantiate this class with execution "sequential" or "vectorized" but later call it with
    "parallel", parallelization will happen on the device that the models were loaded on.
    """

    def __init__(
        self,
        results_path: Union[str, Path],
        estimator: str,
        device: Optional[str] = None,
        execution: ExecutionMode = "sequential",
        num_workers: int = 4,
        n_gpus: int = 1,
    ) -> None:
        super().__init__()
        _check_execution(execution)

        self.results_path = Path(results_path).resolve()
        self.estimator_name = estimator
        self.execution: ExecutionMode = execution
        self.num_workers = num_workers

        self.config: MainConfig = load_config(self.results_path / "config.yaml")
        self.estimator_config: EstimatorConfig = self.config.estimators[estimator]
        self.snapshot: _EstimatorSnapshot = load_snapshot(self.results_path, estimator)

        self.models = nn.ModuleDict()
        self._scores: Dict[str, Dict[str, float]] = {}
        self._vectorized_groups = nn.ModuleDict()
        devices = _resolve_devices(device, n_gpus if execution == "parallel" else 1)
        self._load_models(devices)

    @property
    def device(self) -> torch.device:
        """Where inputs go and outputs come back to: the device of the first loaded model."""
        return _module_device(next(iter(self.models.values())))

    def _leaves(self, systematics: List[str]) -> List[_Leaf]:
        """All leaves of `systematics`, in a fixed sorted order shared by loading, vectorization and
        aggregation.
        """
        return [
            _Leaf(systematic, ensemble, fold)
            for systematic in systematics
            for ensemble, folds in sorted(self.snapshot[systematic].items())
            for fold in sorted(folds)
        ]

    def _load_models(self, devices: List[torch.device]) -> None:
        logger.info(f"Loading models for estimator {self.estimator_name!r} onto device(s): {devices}")

        systematic_configs = {
            systematic: merge_systematic_config(self.estimator_config, systematic) for systematic in self.snapshot
        }

        for idx, leaf in enumerate(self._leaves(list(self.snapshot))):
            systematic_config = systematic_configs[leaf.systematic]
            self.models[leaf.key], self._scores[leaf.key] = self._load_single_model(
                systematic_config.model_override,
                systematic_config.dataset_override,
                self.snapshot[leaf.systematic][leaf.ensemble][leaf.fold],
                devices[idx % len(devices)],
            )

        logger.info(f"Loaded {len(self.models)} models")

    def _load_single_model(
        self,
        model_config: Any,
        dataset_config: Any,
        ckpt_path: str,
        device: torch.device,
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Load one checkpoint into a frozen, eval-mode model on `device`, together with the scores
        its training recorded (see `_checkpoint_scores`).
        """
        lightning_module = hydra_instantiate(model_config, dataset_config=dataset_config)

        if hasattr(lightning_module, "configure_model"):
            lightning_module.configure_model()

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        lightning_module.load_state_dict(_clean_state_dict(state_dict, lightning_module))

        # Unwrap the Lightning module so `forward()` matches the raw model's signature.
        if isinstance(lightning_module, L.LightningModule) and hasattr(lightning_module, "model"):
            if not isinstance(lightning_module.model, torch.nn.Module):
                raise ValueError(f"LightningModule {lightning_module} does not wrap around a torch.nn.Module")

            model = lightning_module.model
        else:
            model = lightning_module  # fallback in case lightning_module actually torch module

        model.eval().to(device)

        for param in model.parameters():
            param.requires_grad = False

        return model, _checkpoint_scores(checkpoint)

    def _vectorized_group(self, systematic: str) -> _VectorizedSiblings:
        """Build (and cache) the `_VectorizedSiblings` batching every ensemble/fold sibling under
        one systematic.
        """
        if systematic not in self._vectorized_groups:
            keys = [leaf.key for leaf in self._leaves([systematic])]
            self._vectorized_groups[systematic] = _VectorizedSiblings(
                keys, [self.models[key] for key in keys], self.device
            )

        return self._vectorized_groups[systematic]  # type: ignore[return-value]

    def _compute_leaf_outputs(
        self,
        x: torch.Tensor,
        systematics: List[str],
        execution: ExecutionMode,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate every leaf. The result is a flat `{checkpoint_key: output}` map consumed by
        the fold -> ensemble -> systematic aggregation tree in `forward`.
        """
        keys = [leaf.key for leaf in self._leaves(systematics)]

        match execution:
            case "vectorized":
                outputs: Dict[str, torch.Tensor] = {}

                for systematic in systematics:
                    outputs.update(self._vectorized_group(systematic)(x))

                return outputs

            case "sequential":
                return {key: self.models[key](x) for key in keys}

            case "parallel":
                def _call(key: str) -> torch.Tensor:
                    model = self.models[key]

                    with torch.no_grad():
                        return model(x.to(_module_device(model))).to(x.device)

                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    return dict(zip(keys, executor.map(_call, keys)))

    @staticmethod
    def _sibling_metrics(nodes: List[_Node], spec: AggregationSpec) -> Optional[List[float]]:
        """The per-sibling metric named by `spec.metric_key`, handed to the aggregator as `metrics`.

        "best" requires it, any other method (builtin or custom) gets it whenever `metric_key` is set,
        and None otherwise. A lone sibling is passed through, so needs none.
        """
        if len(nodes) == 1 or (spec.metric_key is None and spec.method != "best"):
            return None

        metrics = [node.scores.get(spec.metric_key) for node in nodes] if spec.metric_key else [None]
        if any(metric is None for metric in metrics):
            raise ValueError(
                f"Aggregation {spec.method!r} needs `metric_key` to name a metric monitored by a "
                f"ModelCheckpoint in every checkpoint, got {spec.metric_key!r}."
            )

        return metrics  # type: ignore[return-value]

    def _aggregate(self, nodes: List[_Node], spec: AggregationSpec) -> Tuple[_Node, torch.Tensor]:
        """Combine sibling `nodes` into their parent node and the spread (std) between them, following `spec`."""
        output, std = aggregate_siblings(
            [node.output for node in nodes],
            method=spec.method,
            metrics=self._sibling_metrics(nodes, spec),
            metric_key=spec.metric_key,
        )
        scores = {
            name: min(node.scores[name] for node in nodes)
            for name in nodes[0].scores
            if all(name in node.scores for node in nodes)
        }
        return _Node(output, scores), std

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        systematics_keys: Optional[List[str]] = None,
        execution: Optional[ExecutionMode] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and aggregate folds -> ensembles -> systematics.

        Args:
            x: Input tensor, forwarded to every fold model.
            systematics_keys: Restrict aggregation to these systematic keys (default: all trained).
            execution: Override the instance's `execution` mode for this call only (see the class
                docstring). Defaults to `self.execution`.

        Returns:
            (mean, std): The aggregated estimator output and its cross-model uncertainty (for the
                outer-most layer, intermediate uncertainties are not propagated).
        """
        execution = execution or self.execution
        _check_execution(execution)

        systematics = [s for s in self.snapshot if systematics_keys is None or s in systematics_keys]
        if not systematics:
            raise ValueError(f"No systematics matched {systematics_keys!r}. Available are: {list(self.snapshot)}")

        outputs = self._compute_leaf_outputs(x.to(self.device), systematics, execution)
        leaf_nodes = {key: _Node(output, self._scores[key]) for key, output in outputs.items()}

        fold_spec = self.estimator_config.expands.folds.aggregation
        ensemble_spec = self.estimator_config.expands.ensembles.aggregation
        systematic_spec = self.estimator_config.systematic_aggregation

        systematic_nodes: List[_Node] = []
        for systematic in systematics:
            ensemble_nodes = []

            for ensemble, folds in sorted(self.snapshot[systematic].items()):
                fold_nodes = [leaf_nodes[_Leaf(systematic, ensemble, fold).key] for fold in sorted(folds)]
                ensemble_nodes.append(self._aggregate(fold_nodes, fold_spec)[0])

            systematic_nodes.append(self._aggregate(ensemble_nodes, ensemble_spec)[0])

        mean, std = self._aggregate(systematic_nodes, systematic_spec)
        return mean.output, std


__all__ = [
    "Estimator",
    "load_snapshot",
]
