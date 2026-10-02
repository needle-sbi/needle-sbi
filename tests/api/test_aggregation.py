"""Tests for needle/api/aggregation.py."""

from __future__ import annotations

import pytest
import torch

from needle.api.aggregation import aggregate_siblings
from needle.utils.config_schema import AggregationSpec


def _weighted_mean(
    outputs: list[torch.Tensor],
    metrics: list[float] | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Example user-defined aggregator, wired in via a dotted `AggregationSpec.method` path,
    implementing the `needle.api.aggregation.Aggregator` protocol. NEEDLE has no built-in "weighted_mean"
    and `AggregationSpec` carries no generic `weights` field on purpose: a custom aggregator that
    wants weights just captures them itself (a module-level constant here; a real project would
    more likely read them off its own config, or use `functools.partial`) instead of routing them
    through the framework.

    A real project would put this in its own installed package rather than a test module.
    """
    weights = [3.0, 1.0]  # arbitrary, hardcoded for this example
    stacked = torch.stack(outputs, dim=0)
    weight_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
    weight_tensor = weight_tensor / weight_tensor.sum()
    view_shape = (-1,) + (1,) * (stacked.dim() - 1)

    aggregated = (weight_tensor.view(view_shape) * stacked).sum(dim=0)
    std = torch.sqrt((weight_tensor.view(view_shape) * (stacked - aggregated) ** 2).sum(dim=0))
    return aggregated, std


class TestAggregate:
    def test_mean_is_the_default(self) -> None:
        outputs = [torch.ones(2, 1), torch.zeros(2, 1)]
        mean, std = aggregate_siblings(outputs)
        assert torch.allclose(mean, torch.full((2, 1), 0.5))
        assert torch.allclose(std, torch.stack(outputs, dim=0).std(dim=0))

    def test_sum(self) -> None:
        outputs = [torch.ones(2, 1), torch.ones(2, 1)]
        total, _ = aggregate_siblings(outputs, method="sum")
        assert torch.allclose(total, torch.full((2, 1), 2.0))

    def test_sum_std_is_per_event_and_independent_of_batch(self) -> None:
        # Events with identical sibling spread must get identical std, whatever else is in the batch.
        outputs = [torch.tensor([[0.0], [0.0]]), torch.tensor([[2.0], [10.0]])]
        _, std = aggregate_siblings(outputs, method="sum")
        spread = torch.stack(outputs, dim=0).std(dim=0)
        assert torch.allclose(std, spread * 2**0.5)
        assert not torch.allclose(std[0], std[1])

    def test_best_selects_lowest_metric(self) -> None:
        outputs = [torch.full((2, 1), 10.0), torch.full((2, 1), 20.0)]
        best, std = aggregate_siblings(outputs, method="best", metrics=[0.5, 0.1])
        assert torch.allclose(best, torch.full((2, 1), 20.0))
        assert torch.allclose(std, torch.zeros(2, 1))

    def test_best_requires_metrics(self) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        with pytest.raises(ValueError, match="metrics required"):
            aggregate_siblings(outputs, method="best")

    def test_custom_aggregator_implements_weighted_mean(self) -> None:
        # NEEDLE has no built-in "weighted_mean" method; this shows how to add one yourself via a
        # dotted-path `method`, resolved by `aggregate_siblings`. See `_weighted_mean` above and
        # docs/concepts/hydra_config.md for the same example.
        outputs = [torch.zeros(2, 1), torch.full((2, 1), 4.0)]
        mean, _ = aggregate_siblings(outputs, method="tests.api.test_aggregation._weighted_mean")
        assert torch.allclose(mean, torch.full((2, 1), 1.0))

    def test_custom_aggregator_missing_attribute_raises_unknown_method(self) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_siblings(outputs, method="tests.api.test_aggregation._does_not_exist")

    def test_single_output_is_passthrough(self) -> None:
        output = torch.full((2, 1), 3.0)
        mean, std = aggregate_siblings([output])
        assert torch.equal(mean, output)
        assert torch.equal(std, torch.zeros_like(output))

    def test_unknown_method_raises(self) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_siblings(outputs, method="median")

    def test_config_driven_spec_forwards_via_call_signature(self) -> None:
        # This is how `Estimator.forward` calls `aggregate_siblings`: reading `method`/`metric_key`
        # off an `AggregationSpec` straight from the resolved config.
        outputs = [torch.ones(2, 1), torch.zeros(2, 1)]
        spec = AggregationSpec(method="mean")
        mean, _ = aggregate_siblings(outputs, method=spec.method, metric_key=spec.metric_key)
        assert torch.allclose(mean, torch.full((2, 1), 0.5))
