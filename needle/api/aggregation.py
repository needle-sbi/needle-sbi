"""Combining sibling predictions (folds, ensemble members, systematic variations) into one.

Used by `needle.api.eval.Estimator`, and the extension point for custom aggregators: any callable
matching the `Aggregator` protocol can be named as a dotted `AggregationSpec.method` path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from hydra.utils import get_method


class Aggregator(Protocol):
    """The definition of what a custom aggregation callable must look like.

    A dotted `AggregationSpec.method` path is resolved to a callable matching this signature. The
    `metrics` holds the per-sibling values of `AggregationSpec.metric_key` (None if unset). It is required
    for "best"; "mean" and "sum" ignore it, and a custom aggregator may use it.
    """

    def __call__(
        self,
        outputs: List[torch.Tensor],
        metrics: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


def _mean(
    outputs: List[torch.Tensor], metrics: Optional[List[float]] = None, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(outputs, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)


def _sum(
    outputs: List[torch.Tensor], metrics: Optional[List[float]] = None, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Siblings are independent estimates, so the sum of their spread is then sqrt(n) * spread."""
    stacked = torch.stack(outputs, dim=0)
    return stacked.sum(dim=0), stacked.std(dim=0) * math.sqrt(len(outputs))


def _best(
    outputs: List[torch.Tensor], metrics: Optional[List[float]] = None, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best means lowest metric (akin to loss functions)"""
    if metrics is None:
        raise ValueError("metrics required for 'best' aggregation")

    best = outputs[int(torch.tensor(metrics).argmin())]
    return best, torch.zeros_like(best)


_BUILTIN_AGGREGATORS: Dict[str, Aggregator] = {"mean": _mean, "sum": _sum, "best": _best}


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
            aggregation callable matching the `Aggregator` protocol.
            Call as `aggregate_siblings(outputs, method=spec.method, metric_key=spec.metric_key)` if
            using the AggregationSpec schema from the config.
        metrics: Per-sibling validation metric, required for `method == "best"` but optional otherwise.
        **kwargs: Forwarded to a custom aggregator (e.g. `metric_key`, or any other field
            your own `AggregationSpec` carries). Ignored by the built-in methods, which need nothing
            beyond `outputs`/`metrics`. There is no generic `weights` mechanism, a custom aggregator
            that needs weights must obtain them from elsewhere rather than routing them through this function.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (aggregated, std) where `aggregated` is the merged result
            and `std` is the spread across siblings (zero for "best").

    The Tensors are first stacked around the outer dimension (`dim=0`), then aggregated according to
    the method. Supported built-in methods (matching available keys in ``AggregationSpec``) are:
        - "mean": `mean()`
        - "sum": `sum()`, with std `sqrt(n_siblings) * spread` of the siblings
        - "best": `outputs[metrics.argmin()]` (lower metric is better)
    Anything else is resolved as a dotted path to a user-supplied callable (see `Aggregator`); see
    ``docs/concepts/hydra_config.md`` for a worked example.
    """
    if len(outputs) == 1:
        return outputs[0], torch.zeros_like(outputs[0])

    aggregator = _BUILTIN_AGGREGATORS.get(method)
    if aggregator is None:
        try:
            aggregator = get_method(method)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ValueError(f"Unknown aggregation method: {method}") from exc

    return aggregator(outputs, metrics=metrics, **kwargs)


__all__ = [
    "Aggregator",
    "aggregate_siblings",
]
