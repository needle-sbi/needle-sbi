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


def _recording_aggregator(
    outputs: list[torch.Tensor],
    metrics: list[float] | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom aggregator that reports what `aggregate_siblings` forwarded to it through its return value.

    (A module-level call log would not work: hydra imports this module a second time by dotted path.)
    The mean is the received `metrics` (``[-1]`` for None) and the std is the number of received kwargs.
    """
    received = torch.tensor(metrics if metrics is not None else [-1.0])
    return received, torch.tensor([float(len(kwargs))])


class TestCustomAggregatorForwarding:
    """Hardening: a dropped `metrics=`/`**kwargs` in `aggregate_siblings` was a real regression."""

    METHOD = "tests.api.test_aggregation._recording_aggregator"

    def test_metrics_and_metric_key_reach_custom_aggregator(self) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        metrics, n_kwargs = aggregate_siblings(outputs, method=self.METHOD, metrics=[0.5, 0.1], metric_key="val_loss")
        assert metrics.tolist() == pytest.approx([0.5, 0.1])
        assert n_kwargs.item() == 1.0

    def test_metrics_default_to_none_for_custom_aggregator(self) -> None:
        metrics, n_kwargs = aggregate_siblings([torch.zeros(1), torch.ones(1)], method=self.METHOD)
        assert metrics.tolist() == [-1.0]
        assert n_kwargs.item() == 0.0

    def test_single_sibling_never_calls_custom_aggregator(self) -> None:
        output = torch.zeros(1)
        result, _ = aggregate_siblings([output], method=self.METHOD, metrics=[0.1])
        assert torch.equal(result, output)


class TestAggregateAttacks:
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_integer_outputs_give_a_clear_error_or_a_result(self, method: str) -> None:
        # e.g. a classifier returning argmax labels: `Tensor.mean` on int64 raises an opaque dtype error.
        outputs = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        try:
            aggregated, _ = aggregate_siblings(outputs, method=method)
        except (ValueError, TypeError, RuntimeError) as exc:
            assert "dtype" in str(exc) or "floating" in str(exc)
        else:
            assert aggregated.tolist() == ([2.0, 3.0] if method == "mean" else [4, 6])

    def test_empty_outputs_raise_a_clear_error(self) -> None:
        with pytest.raises((ValueError, RuntimeError), match="non-empty|at least one|empty"):
            aggregate_siblings([])

    def test_mismatched_sibling_shapes_raise_a_clear_error(self) -> None:
        with pytest.raises((ValueError, RuntimeError), match="equal size|shape"):
            aggregate_siblings([torch.zeros(3, 1), torch.zeros(4, 1)])

    def test_best_ignores_a_nan_metric_instead_of_selecting_it(self) -> None:
        # A diverged run has val_loss = nan; `argmin` treats nan as the minimum, so it would win.
        outputs = [torch.full((2, 1), 1.0), torch.full((2, 1), 2.0)]
        best, _ = aggregate_siblings(outputs, method="best", metrics=[float("nan"), 0.3])
        assert torch.allclose(best, torch.full((2, 1), 2.0)), "'best' selected the sibling whose metric is NaN"

    @pytest.mark.parametrize("metrics", [[0.1], [0.3, 0.2, 0.1]])
    def test_best_rejects_metrics_of_the_wrong_length(self, metrics: list[float]) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        with pytest.raises(ValueError, match="metrics"):
            aggregate_siblings(outputs, method="best", metrics=metrics)

    def test_best_ties_pick_the_first_sibling(self) -> None:
        outputs = [torch.full((1,), 1.0), torch.full((1,), 2.0)]
        best, _ = aggregate_siblings(outputs, method="best", metrics=[0.2, 0.2])
        assert best.item() == 1.0

    def test_single_sibling_with_unknown_method_still_raises(self) -> None:
        # A typo in `method` must not be hidden just because only one sibling happens to exist.
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_siblings([torch.zeros(2, 1)], method="medain")

    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_does_not_modify_or_alias_inputs(self, method: str) -> None:
        outputs = [torch.rand(3, 1), torch.rand(3, 1)]
        before = [o.clone() for o in outputs]
        aggregated, std = aggregate_siblings(outputs, method=method)
        aggregated += 1
        std += 1
        assert all(torch.equal(o, b) for o, b in zip(outputs, before))

    def test_passthrough_and_best_do_not_alias_inputs(self) -> None:
        # In-place edits of the returned prediction (e.g. `mean -= offset`) must not corrupt
        # the sibling that the caller (or `Estimator`) still holds.
        single = torch.zeros(2, 1)
        result, _ = aggregate_siblings([single])
        result += 1
        assert torch.equal(single, torch.zeros(2, 1)), "single-sibling passthrough aliases its input"

        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        best, _ = aggregate_siblings(outputs, method="best", metrics=[0.1, 0.5])
        best += 5
        assert torch.equal(outputs[0], torch.zeros(2, 1)), "'best' returns the winning input tensor itself"

    @pytest.mark.parametrize("method, metrics", [("mean", None), ("sum", None), ("best", [0.1, 0.2])])
    def test_zero_events_give_zero_events(self, method: str, metrics: list[float] | None) -> None:
        outputs = [torch.zeros(0, 2), torch.zeros(0, 2)]
        aggregated, std = aggregate_siblings(outputs, method=method, metrics=metrics)
        assert aggregated.shape == std.shape == (0, 2)

    def test_nan_in_one_event_does_not_contaminate_other_events(self) -> None:
        outputs = [torch.tensor([[1.0], [float("nan")]]), torch.tensor([[3.0], [4.0]])]
        mean, std = aggregate_siblings(outputs)
        assert mean[0].item() == 2.0 and torch.isfinite(std[0])
        assert torch.isnan(mean[1])

    def test_float64_inputs_stay_float64(self) -> None:
        outputs = [torch.ones(2, 1, dtype=torch.float64), torch.zeros(2, 1, dtype=torch.float64)]
        for method in ("mean", "sum"):
            aggregated, std = aggregate_siblings(outputs, method=method)
            assert aggregated.dtype == std.dtype == torch.float64

    def test_sum_std_for_three_siblings_scales_with_sqrt_n(self) -> None:
        outputs = [torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0])]
        _, std = aggregate_siblings(outputs, method="sum")
        assert std.item() == pytest.approx(1.0 * 3**0.5)
