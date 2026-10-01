from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union
from urllib.parse import parse_qsl

import torch
import torch.nn as nn
from hydra.utils import get_method
from torch.func import functional_call, vmap

from needle.api.config import load_config
from needle.utils.config_schema import EstimatorConfig, MainConfig
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


class Aggregator(Protocol):
    """The one, formal definition of what a custom aggregation callable must look like.

    A dotted `AggregationSpec.method` path is resolved to a callable matching this signature -
    there's no separate prose description of the signature to keep in sync elsewhere; `aggregate_siblings`
    itself satisfies this protocol for its built-in "mean"/"sum"/"best" methods too.
    """

    def __call__(
        self,
        outputs: List[torch.Tensor],
        metrics: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


def aggregate_siblings(
    outputs: List[torch.Tensor],
    method: str = "mean",
    metrics: Optional[List[float]] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine sibling predictions into a single Tensor. A sibling can be either `systematic`,
    `ensemble` or `fold`.

    Args:
        outputs: One prediction Tensor per sibling, in the same order as `metrics`.
        method: One of the built-ins "mean" / "sum" / "best", or a dotted import path to a custom
            aggregation callable matching the `Aggregator` protocol. Equivalent to
            `AggregationSpec.method`; call as `aggregate_siblings(outputs, method=spec.method,
            metric_key=spec.metric_key)` from a config-driven `AggregationSpec`.
        metrics: Per-sibling validation metric, required for `method == "best"`.
        **kwargs: Forwarded verbatim to a custom aggregator (e.g. `metric_key`, or any other field
            your own `AggregationSpec` carries). Ignored by the built-in methods, which need nothing
            beyond `outputs`/`metrics`. There is no generic `weights` mechanism: a custom aggregator
            that needs weights captures them itself rather than routing them through this function.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (aggregated, std) where `aggregated` is the merged result
            and `std` is the spread across siblings (zero for "best").

    The Tensors are first stacked around the outer dimension (`dim=0`), then aggregated according to
    the method. Supported built-in methods (matching available keys in ``AggregationSpec``) are:
        - "mean": `mean()`
        - "sum": `sum()`
        - "best": `outputs[metrics.argmin()]` (lower metric is better)
    Anything else is resolved as a dotted path to a user-supplied callable (see `Aggregator`); see
    ``docs/concepts/hydra_config.md`` for a worked example (a weighted mean implemented as a custom
    aggregator).
    """
    if len(outputs) == 1:
        return outputs[0], torch.zeros_like(outputs[0])

    stacked = torch.stack(outputs, dim=0)

    if method == "mean":
        return stacked.mean(dim=0), stacked.std(dim=0)

    if method == "sum":
        variances = stacked.var(dim=0)
        std = torch.sqrt(variances.sum(dim=0, keepdim=True).expand_as(variances))
        return stacked.sum(dim=0), std

    if method == "best":
        if metrics is None:
            raise ValueError("metrics required for 'best' aggregation")

        best_idx = int(torch.tensor(metrics).argmin())
        return outputs[best_idx], torch.zeros_like(outputs[best_idx])

    try:
        aggregator: Aggregator = get_method(method)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f"Unknown aggregation method: {method}") from exc

    return aggregator(outputs, metrics=metrics, **kwargs)


ExecutionMode = Literal["sequential", "parallel", "vectorized"]
_EXECUTION_MODES = ("sequential", "parallel", "vectorized")


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
        self._template = models[0]

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
            return functional_call(self._template, params, (x,))

        batched = vmap(model_fn, in_dims=(0, None))(params, x)
        return {key: batched[i] for i, key in enumerate(self.keys)}


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
    dotted path to a custom callable (see `needle.aggregate_siblings`).

    The leaf models (one per systematic/ensemble/fold combination) can be evaluated in one of
    three ``execution`` modes, settable at construction and overridable per-call via
    `forward(..., execution=...)`:

        - "sequential" (default): one Python-level forward call per leaf model. Simplest, always
          correct, no extra memory overhead.

        - "parallel": every leaf model (across all systematics/ensembles/folds at once)
          is submitted to a `ThreadPoolExecutor` with ``num_workers`` threads.
          PyTorch releases the GIL inside its kernels, so this overlaps Python/launch overhead.
          When constructed with ``n_gpus > 1``, runs the models of different devices concurrently.
          Models on the same device still share that device's stream, and on the CPU the threads
          compete for PyTorch's intra-op thread pool, so expect gains mainly for many small models
          or several devices. Works on any accelerator, not only CUDA.

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

        # Leaf models only get spread across several devices when "parallel" is requested at
        # construction time, since that's when model placement happens (`_load_models`); a
        # later per-call `execution="parallel"` override on a "sequential"/"vectorized" instance
        # still parallelizes, just on whichever single device the models were already loaded to.
        self._devices = _resolve_devices(device, n_gpus if execution == "parallel" else 1)
        self.device = self._devices[0]

        self.config: MainConfig = load_config(self.results_path / "config.yaml")
        self.estimator_config: EstimatorConfig = self.config.estimators[estimator]
        self.snapshot: _EstimatorSnapshot = load_snapshot(self.results_path, estimator)

        self.models = nn.ModuleDict()
        self._model_devices: Dict[str, torch.device] = {}
        self._vectorized_groups = nn.ModuleDict()
        self._load_models()

    @staticmethod
    def _checkpoint_key(systematic: str, ensemble: int, fold: int) -> str:
        return f"syst={systematic}&ensem={ensemble}&fold={fold}"

    def _leaves(self, systematics: List[str]) -> List[Tuple[str, int, int, str]]:
        """All ``(systematic, ensemble, fold, checkpoint_key)`` leaves of `systematics`, in a fixed
        sorted order shared by loading, vectorization and aggregation.
        """
        return [
            (systematic, ensemble, fold, self._checkpoint_key(systematic, ensemble, fold))
            for systematic in systematics
            for ensemble, folds in sorted(self.snapshot[systematic].items())
            for fold in sorted(folds)
        ]

    def _load_models(self) -> None:
        logger.info(f"Loading models for estimator {self.estimator_name!r} onto device(s): {self._devices}")

        systematic_configs = {
            systematic: merge_systematic_config(self.estimator_config, systematic) for systematic in self.snapshot
        }

        for idx, (systematic, ensemble, fold, key) in enumerate(self._leaves(list(self.snapshot))):
            systematic_config = systematic_configs[systematic]
            target_device = self._devices[idx % len(self._devices)]
            self._model_devices[key] = target_device
            self.models[key] = self._load_single_model(
                systematic_config.model_override,
                systematic_config.dataset_override,
                self.snapshot[systematic][ensemble][fold],
                target_device,
            )

        logger.info(f"Loaded {len(self.models)} models")

    def _load_single_model(
        self, model_config: Any, dataset_config: Any, ckpt_path: str, device: torch.device
    ) -> nn.Module:
        model = hydra_instantiate(model_config, dataset_config=dataset_config)

        if hasattr(model, "configure_model"):
            model.configure_model()

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(_clean_state_dict(state_dict, model))

        # Unwrap the Lightning module so `forward()` matches the raw model's signature.
        if hasattr(model, "model"):
            model = model.model

        model.eval().to(device)

        for param in model.parameters():
            param.requires_grad = False

        return model

    def _vectorized_group(self, systematic: str) -> _VectorizedSiblings:
        """Build (and cache) the `_VectorizedSiblings` batching every ensemble/fold sibling under
        one systematic.
        """
        if systematic not in self._vectorized_groups:
            keys = [key for *_, key in self._leaves([systematic])]
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
        if execution == "vectorized":
            outputs: Dict[str, torch.Tensor] = {}

            for systematic in systematics:
                outputs.update(self._vectorized_group(systematic)(x))

            return outputs

        keys = [key for *_, key in self._leaves(systematics)]

        if execution == "sequential":
            return {key: self.models[key](x) for key in keys}

        # "parallel": models are mutually independent, and each runs on the device it was loaded to.
        inputs = {
            device: x
            if device == self.device
            else x.to(device)
            for device in set(self._model_devices.values())
        }

        def _call(key: str) -> torch.Tensor:
            with torch.no_grad():
                output = self.models[key](inputs[self._model_devices[key]])

            return output if output.device == self.device else output.to(self.device)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            return dict(zip(keys, executor.map(_call, keys)))

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

        fold_spec = self.estimator_config.expands.folds.aggregation
        ensemble_spec = self.estimator_config.expands.ensembles.aggregation

        leaf_outputs = self._compute_leaf_outputs(x.to(self.device), systematics, execution)

        systematic_outputs: List[torch.Tensor] = []
        for systematic in systematics:
            ensemble_outputs = []

            for ensemble, folds in sorted(self.snapshot[systematic].items()):
                fold_outputs = [
                    leaf_outputs[self._checkpoint_key(systematic, ensemble, fold)] for fold in sorted(folds)
                ]
                aggregated, _ = aggregate_siblings(
                    fold_outputs,
                    method=fold_spec.method,
                    metric_key=fold_spec.metric_key,
                )
                ensemble_outputs.append(aggregated)

            aggregated, _ = aggregate_siblings(
                ensemble_outputs,
                method=ensemble_spec.method,
                metric_key=ensemble_spec.metric_key,
            )
            systematic_outputs.append(aggregated)

        systematic_spec = self.estimator_config.systematic_aggregation
        mean, std = aggregate_siblings(
            systematic_outputs,
            method=systematic_spec.method,
            metric_key=systematic_spec.metric_key,
        )
        return mean, std


__all__ = [
    "Aggregator",
    "Estimator",
    "load_snapshot",
    "aggregate_siblings",
]
