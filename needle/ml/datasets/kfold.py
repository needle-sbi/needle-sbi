"""
Small helper functions for ML Datasets
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("ml")

#: Mapping from fold index to an optional partition size override.
type PartitionDict = dict[int, Optional[int]]

#: Dask division boundaries expressed as a sorted tuple of integer row offsets.
type Divisions = tuple[int, ...]


class KFold:
    """Take care of k-Fold training for chunked datasets."""

    def __init__(
        self,
        fold_index: int,
        n_folds: int,
        divisions: Divisions | tuple,
        is_training: bool,
    ):
        """Register a new KFold Manager

        This class implements functions to mask and shuffle partitions in order to work with
        :class:`torch.IterableDataset`.

        This k-fold training class uses the rolling window method to determine which partitions are
        kept inside a fold and which are discarded. In order to ensure accurate slicing even for
        unequally sized partitions, the partitions found at the boundary between folds are sliced.

        Note:
            No actual data is computed inside this class, only the management and assignment of partitions
            and eventual slicing indices.

        Args:
            fold_index (int): The current fold index. Has to be within the bounds of possible indices
            n_folds (int): The total number of folds
            divisions (tuple): Tuple of npartitions + 1 values, in ascending order, marking the lower/upper
                bounds of each partition's index. (Source: dask Dataframe documentation).
            is_training (int, optional): Whether the mixed-in class is associated to training or to
                validation. This determines which indices are returned. If True, the kfold_indices()
                method will return all indices except those for validation (all folds except one).
                If False, return the validation indices (one fold).

        Example:
            Finding the divisions of an Array. For a generic :class:`dask_awkward.Array`, one can use
            the following method

            >>> array = dak.from_awkward(ak.Array(np.arange(10_000)), 100)
            >>> array.eager_compute_divisions()
            >>> array.divisions
            (0, 100, 200, 300, ..., 10000)

        """
        self.fold_index = fold_index
        self.n_folds = n_folds
        self.divisions = divisions
        self.is_training = is_training
        self._keep, self._drop = self._kfold_indices()
        self._partition_dict: PartitionDict = self._keep if is_training else self._drop

    @property
    def partition_ids(self) -> list[int]:
        """List of valid partition ids

        Useful for checking if a partition should be included in the training data and constructing
        a mask of partition_ids for :meth:`KFold.mask`.

        Returns:
            list[int]: Partition IDs, e.g. [0, 1, ... 5]
        """
        return list(self._partition_dict.keys())

    @property
    def partitions(self) -> PartitionDict:
        """Dictionary with valid partitions IDs and potential slicing indices

        Returns:
            PartitionDict: A dict of format `{<partition_id>: <slicing_index>}`.

                The slicing index is one of three cases

                - None if the corresponding partition is fully within the current fold. In this
                    case the whole partition should be loaded from disk without slicing.

                - int > 0 if there is an upper cutoff to the local index inside the partitions. If
                    the partition has length 10k and this slicing_index is 7k, one should slice the
                    array before computing with `array.partitions[<partition_id>][0:<slicing_index>]`.
                    This is the case when the fold boundary starts inside this partition.

                - int < 0 if it is a lower cutoff instead. Here one has to slice from below and remove
                    all events below this threshold
                    `array.partitions[<partition_id>][<slicing_index>:-1]`.
        """
        return self._partition_dict

    def mask(self, mask: list[bool]) -> None:
        """Mask away partitions based on a boolean mask

        Args:
            mask (list[bool]): A list of booleans of the same length as the current
                :attr:`KFold.partitions`.

        Raises:
            IndexError: If the lengths of the mask and the current valid partitions
                (:attr:`KFold.partitions`) do not match.

        Example:
            If you are using torch multiprocessing and want to keep only a subset of partitions for
                each workers.
            Assuming you are working with :class:`torch.utils.data.IterableDataset`

            >>> worker_info = torch.utils.data.get_worker_info()
            ... kfold = KFold(...)
            ... assigned_partition_mask = [
            ...     p % worker_info.num_workers == self.worker_info.id for p in kfold.partition_ids
            ... ]
            ... kfold.mask(assigned_partition_mask)
        """
        if len(mask) != len(self._partition_dict):
            raise IndexError(
                f"Length of mask {len(mask)} does not match that of available partitions {len(self._partition_dict)}"
            )
        self._partition_dict = {pid: value for i, (pid, value) in enumerate(self._partition_dict.items()) if mask[i]}

    def shuffle(self, rng: np.random.Generator | None = None) -> None:
        """Shuffle the partitions using :meth:`numpy.permutation`.

        Args:
            rng (np.random.Generator, optional): A Generator to allow reproducible results with the same seed.
        """
        rng = rng or np.random.default_rng()
        shuffled = rng.permutation(np.array(list(self._partition_dict.items())))
        self._partition_dict = dict(shuffled.tolist())  # type: ignore

    def _kfold_indices(self) -> tuple[PartitionDict, PartitionDict]:
        """Produce training-eval split for partitioned datasets

        Returns:
            tuple: A 2-Tuple of the training partitions (K-1 folds) and the evaluation partitions (1 fold).
                For a better description of the `PartitionDict` format, see :attr:`KFold.partitions` docstring.
        """
        if not any(self.divisions):
            raise ValueError(
                f"Divisions are unknown, {self.divisions=}. Make sure to use `dask_awkward.eager_compute_divisions` "
                "to get a valid set of divisions."
            )
        if self.fold_index >= self.n_folds:
            raise IndexError(
                f"Fold index {self.fold_index} is outside the available range of folds, (0, ..., {self.n_folds - 1})"
            )

        fold_start, fold_end = self._fold_boundaries(self.divisions[-1], self.n_folds, self.fold_index)  # as event

        partition_id_of_fold_start = self._find_partition_of_event(self.divisions, fold_start)  # as partition ID
        partition_id_of_fold_end = self._find_partition_of_event(self.divisions, fold_end)

        keep: dict[int, Optional[int]] = {}
        drop: dict[int, Optional[int]] = {}

        for pid in range(len(self.divisions) - 1):
            partition_event_start = fold_start - self.divisions[pid]  # local event index
            partition_event_end = fold_end - self.divisions[pid + 1]

            if pid == partition_id_of_fold_start:
                if partition_event_start == 0:
                    drop[pid] = None
                else:
                    keep[pid] = partition_event_start
                    drop[pid] = -partition_event_start
            elif pid == partition_id_of_fold_end:
                if partition_event_end == 0:
                    drop[pid] = None
                else:
                    keep[pid] = -partition_event_end
                    drop[pid] = partition_event_end
            else:
                if (self.divisions[pid + 1] < fold_start) or (fold_end < self.divisions[pid]):
                    keep[pid] = None
                else:
                    drop[pid] = None

        return keep, drop

    @staticmethod
    def _fold_boundaries(total_events: int, n_folds: int, fold_index: int) -> tuple[int, int]:
        """Compute global (start, end) event indices for a given fold

        Args:
            total_events (int): The total number of events in the whole array. Can be computed with
                the length attribute of :class:`Ingestor` or from :meth:`numpy.diff` function applied
                to the divisions of the array.
            n_folds (int): The number of folds used in this training
            fold_index (int): The current fold index for which to compute the boundaries

        Returns:
            tuple[int, int]: A 2-tuple of start and end index for the whole array.
        """
        fold_size = total_events // n_folds
        fold_start = fold_index * fold_size
        fold_end = total_events if fold_index == n_folds - 1 else (fold_index + 1) * fold_size
        return (fold_start, fold_end)

    @staticmethod
    def _find_partition_of_event(divisions: Divisions, event_index: int) -> int:
        """Return the partition index that contains the given event index

        Args:
            divisions (Divisions): A tuple of int representing the cumulative index of the array
            event_index (int): The event to find inside the divisions of the array

        Returns:
            int: The partition index where the `event_index` resides.

        Note:
            Used to find which partitions host the fold boundaries
        """
        for i in range(len(divisions) - 1):
            if divisions[i] <= event_index < divisions[i + 1]:
                return i
        return len(divisions) - 2
