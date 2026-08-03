import math
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import numpy as np
import pytest

from needle.etl.array import brute_force_divisions
from needle.ml.datasets.io import load_partition
from needle.ml.datasets.kfold import KFold, PartitionDict


@pytest.fixture(scope="module")
def array(tmp_path_factory: pytest.TempPathFactory) -> dak.Array:
    data_dir = tmp_path_factory.mktemp("awkward_data")

    partition_lengths = [
        874_000,
        84_000,
        1_428_000,
        1_369_000,
        1_428_000,
        1_428_000,
        1_003_000,
        1_410_000,
        823_000,
        1_260_000,
    ]

    parquet_files = []

    for i, n in enumerate(partition_lengths):
        arr = ak.Array({"x": np.arange(n)})
        fname = Path(data_dir) / f"part_{i:02d}.parquet"
        ak.to_parquet(arr, fname)
        parquet_files.append(fname)

    dak_array = dak.from_parquet(parquet_files)
    assert isinstance(dak_array, dak.Array)
    dak_array.eager_compute_divisions()

    if not any(dak_array.divisions):
        dak_array._divisions = brute_force_divisions(parquet_files)

    return dak_array


class TestKFold:
    def calculate_array_length(
        self,
        array: dak.Array,
        partition_dict: PartitionDict,
    ) -> int:
        fold_length = 0

        for pid, slicing_index in partition_dict.items():
            sub_array = load_partition(
                array,
                partition_id=pid,
                event_index=slicing_index,
            ).compute()
            fold_length += len(sub_array)

        return fold_length

    @pytest.mark.parametrize("n_folds", list(range(3, 10)))
    def test_instantiation(
        self,
        array: dak.Array,
        n_folds: int,
    ):
        length_fold: dict[int, int] = {}
        desired_ratio = 1 - 1 / n_folds
        assert array.divisions[-1]

        for fold in range(n_folds):
            kfold = KFold(
                fold_index=fold,
                n_folds=n_folds,
                divisions=array.divisions,
                is_training=True,
            )
            length_fold[fold] = self.calculate_array_length(
                array,
                partition_dict=kfold.partitions,
            )
            fold_ratio = length_fold[fold] / array.divisions[-1]
            assert math.isclose(desired_ratio, fold_ratio, rel_tol=0.01)


class TestLoadPartitionNegativeSlice:
    """Regression tests for load_partition's negative event_index branch.

    `array.partitions[pid][abs(event_index):-1]` silently excluded the last event of the
    partition, and produced a zero-length `EmptyArray` (losing all record/field type info)
    whenever exactly one event should have remained after slicing - which then crashed any
    downstream field access (e.g. `array['some_field']`) with `IndexError: cannot slice
    EmptyArray ... not an array of records`. Fixed to `array.partitions[pid][abs(event_index):]`.
    """

    @pytest.fixture
    def ten_event_array(self, tmp_path: Path) -> dak.Array:
        from needle.etl.dask_ingestor import Ingestor

        path = tmp_path / "ten_events.parquet"
        arr = ak.Array({"x": np.arange(10, dtype=np.float32)})
        ak.to_parquet(arr, path)
        return Ingestor(str(path)).array

    @pytest.mark.parametrize("remaining", [5, 2, 1])
    def test_negative_slice_keeps_correct_remaining_events(
        self,
        ten_event_array: dak.Array,
        remaining: int,
    ) -> None:
        event_index = -(10 - remaining)

        sliced = load_partition(
            ten_event_array,
            partition_id=0,
            event_index=event_index,
        ).compute()

        assert len(sliced) == remaining
        assert list(sliced.fields) == ["x"]
        assert sliced["x"].to_list() == list(range(10 - remaining, 10))
