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


class TestPartitionQueuePreservesDivisions:
    """Regression test: PartitionQueue must not silently re-corrupt already-valid divisions.

    `PartitionQueue.__init__` unconditionally called `array.eager_compute_divisions()` again
    whenever `npartitions > 1`, even if the array's divisions had already been correctly
    computed upstream. Re-running dask_awkward's own `eager_compute_divisions()` can
    re-corrupt an already-valid divisions tuple (collapsing interior/trailing boundaries back
    to zero for certain array shapes), which then makes downstream partition slicing (via
    `load_partition`) silently return an empty, typeless `EmptyArray` instead of the real data.

    Uses a multi-file array (like the `array` fixture above) to obtain reliably correct,
    already-valid divisions independent of any single-file/row-group-split repair logic.
    """

    def test_kfold_partitions_are_non_empty_after_partition_queue_construction(
        self,
        array: dak.Array,
    ) -> None:
        from needle.ml.datasets.io import PartitionQueue

        assert array.divisions[0] == 0
        assert all(b > a for a, b in zip(array.divisions, array.divisions[1:]))

        # PartitionQueue construction must not alter already-valid divisions.
        divisions_before = array.divisions
        queue = PartitionQueue(array)
        assert queue.array.divisions == divisions_before

        kfold = KFold(fold_index=0, n_folds=3, divisions=array.divisions, is_training=False)

        total = 0
        for pid, slicing_index in kfold.partitions.items():
            sub_array = queue.load_partition_thread_safe(pid, slicing_index).compute()
            assert len(sub_array) > 0, f"partition {pid} (slice={slicing_index}) was unexpectedly empty"
            total += len(sub_array)

        assert math.isclose(total, array.divisions[-1] / 3, rel_tol=0.01)
