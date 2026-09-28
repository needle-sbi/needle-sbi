from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from urllib.parse import parse_qsl

import torch
import torch.nn as nn
from hydra.utils import get_method
from omegaconf import OmegaConf

from needle.api.config import load_config
from needle.utils.config_schema import EstimatorConfig, MainConfig, SystematicConfig
from needle.utils.config_utils import hydra_instantiate
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
            cleaned = {k[len(prefix) :]: v for k, v in state_dict.items()}
            return cleaned

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


class Estimator(nn.Module):
    """Combined inference model for one trained estimator.

    Loads every checkpoint and aggregates them bottom-up into a single ``(mean, std)`` prediction in
    the following way:

        folds                               (via ``expands.folds.aggregation``)
          └─> ensemble members              (via ``expands.ensembles.aggregation``)
              └─> systematic variations     (via ``estimator.systematic_aggregation``)
                  └─> Estimator

    Each level's `AggregationSpec.method` is one of the built-ins ("mean" / "sum" / "best") or a
    dotted path to a custom callable (see `aggregate_siblings`).

    Examples:
        >>> from needle.api.eval import Estimator
        >>> model = Estimator("runs/my_run", "model_A")
        >>> mean, std = model(x)
    """

    def __init__(
        self,
        results_path: Union[str, Path],
        estimator: str,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.results_path = Path(results_path).resolve()
        self.estimator_name = estimator
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config: MainConfig = load_config(self.results_path / "config.yaml")
        self.estimator_config: EstimatorConfig = self.config.estimators[estimator]
        self.snapshot: _EstimatorSnapshot = load_snapshot(self.results_path, estimator)

        self.models = nn.ModuleDict()
        self._load_models()

    def _systematic_config(self, systematic: str) -> SystematicConfig:
        """Merge the estimator config with its override for one systematic variation, mirroring
        `needle.tasks.base.training.BaseTrainingTask.systematic_config`.
        """
        return OmegaConf.merge(
            OmegaConf.to_container(self.estimator_config.expands.systematics[systematic], resolve=False),
            self.estimator_config,
        )  # type: ignore[return-value]

    @staticmethod
    def _checkpoint_key(systematic: str, ensemble: int, fold: int) -> str:
        return f"syst={systematic}&ensem={ensemble}&fold={fold}"

    def _load_models(self) -> None:
        logger.info(f"Loading models for estimator {self.estimator_name!r} onto device: {self.device}")

        for systematic, ensembles in self.snapshot.items():
            systematic_config = self._systematic_config(systematic)
            model_config = systematic_config.model_override
            dataset_config = systematic_config.dataset_override

            for ensemble, folds in ensembles.items():
                for fold, ckpt_path in folds.items():
                    key = self._checkpoint_key(systematic, ensemble, fold)
                    self.models[key] = self._load_single_model(model_config, dataset_config, ckpt_path)

        logger.info(f"Loaded {len(self.models)} models")

    def _load_single_model(self, model_config: Any, dataset_config: Any, ckpt_path: str) -> nn.Module:
        model = hydra_instantiate(model_config, dataset_config=dataset_config)

        if hasattr(model, "configure_model"):
            model.configure_model()

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(_clean_state_dict(state_dict, model))

        # Unwrap the Lightning module so `forward()` matches the raw model's signature.
        if hasattr(model, "model"):
            model = model.model

        model.eval().to(self.device)

        for param in model.parameters():
            param.requires_grad = False

        return model

    def forward(
        self,
        x: torch.Tensor,
        systematics_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and aggregate folds -> ensembles -> systematics.

        Args:
            x: Input tensor, forwarded to every fold model.
            systematics_keys: Restrict aggregation to these systematic keys (default: all trained).

        Returns:
            (mean, std): The aggregated estimator output and its cross-model uncertainty.
        """
        x = x.to(self.device)
        fold_spec = self.estimator_config.expands.folds.aggregation
        ensemble_spec = self.estimator_config.expands.ensembles.aggregation

        systematic_outputs: Dict[str, torch.Tensor] = {}

        for systematic, ensembles in self.snapshot.items():
            if systematics_keys is not None and systematic not in systematics_keys:
                continue

            ensemble_outputs = []

            for ensemble, folds in sorted(ensembles.items()):
                fold_outputs = [
                    self.models[self._checkpoint_key(systematic, ensemble, fold)](x)
                    for fold, _ in sorted(folds.items())
                ]
                aggregated, _ = aggregate_siblings(
                    fold_outputs, method=fold_spec.method, metric_key=fold_spec.metric_key
                )
                ensemble_outputs.append(aggregated)

            aggregated, _ = aggregate_siblings(
                ensemble_outputs, method=ensemble_spec.method, metric_key=ensemble_spec.metric_key
            )
            systematic_outputs[systematic] = aggregated

        if not systematic_outputs:
            raise ValueError(f"No systematics matched {systematics_keys!r}. Available are: {list(self.snapshot)}")

        if len(systematic_outputs) == 1:
            (only_output,) = systematic_outputs.values()
            return only_output, torch.zeros_like(only_output)

        systematic_spec = self.estimator_config.systematic_aggregation
        return aggregate_siblings(
            list(systematic_outputs.values()),
            method=systematic_spec.method,
            metric_key=systematic_spec.metric_key,
        )


__all__ = [
    "Aggregator",
    "Estimator",
    "load_snapshot",
    "aggregate_siblings",
]
