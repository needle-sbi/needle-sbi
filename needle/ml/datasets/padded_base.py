"""
Base class for padded Datasets. Provided methods for other datasets to inherit from.

TODO Padding happens for each column independently, which improves memory usage. It might
not be finished yet for training and evaluation datasets, as they should have the same
padding.
"""

import logging
import warnings
from abc import ABC, abstractmethod

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

from needle.etl.array import NestedArrayIndexer
from needle.etl.dask_ingestor import Ingestor
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


class PaddedDatasetBase(Dataset, ABC):
    """
    Base class for Datasets handling ragged data from dask_awkward Record arrays.
    """

    SHUFFLE_ALLOWED: bool
    TORCH_MULTIPROCESSING_ALLOWED: bool

    features_ingestor: Ingestor
    labels_ingestor: Ingestor

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def convert_ak_to_tensor(self, array: ak.Array, fields: list[str]) -> torch.Tensor:
        """Compute the tensors from an awkward array.

        This means that any delayed objects such as dask_awkward arrays must be computed before
        calling this method.

        Args:
            array: ak.Array: Awkward Array containing the data.
            fields: list[str]: List of field names to extract from the array.

        Returns:
            torch.Tensor: Tensor of shape (E, P, F) where
                E is the number of events,
                P is the number of particles (padded),
                F is the number of features.
        """
        event_list = []

        for field in fields:
            column = NestedArrayIndexer.get_nested_field(array, field, self.features_ingestor.SEPARATOR)
            column = self.add_innermost_dimension(column)
            padded = ak.pad_none(
                column,
                axis=1,
                target=self.get_padding_length(column),
                clip=True,
            )
            padded = padded[..., np.newaxis]
            event_list.append(padded)

        events = ak.concatenate(event_list, axis=-1)
        events = torch.tensor(ak.to_numpy(events), dtype=torch.float32)
        return events

    def __len__(self) -> int:
        return self.features_ingestor.length

    def get_padding_length(self, array: ak.Array) -> int:
        """Compute the padding length for the features array.

        NOTE:
            - Since 'ak.ravel' is not fully supported by 'dask_awkward', we allow the
                logger to display the warning if the logging level is set to 'DEBUG'.
            - The 'if not hasattr()' check is used for caching the dask result.

        Args:
            array: ak.Array: Awkward Array containing the data.

        Returns:
            int: Padding length at dimension 1 of the given array
        """
        if not hasattr(self, "_feature_padding_length"):

            def _get_length() -> int:
                return int(ak.max(ak.ravel(ak.num(array, axis=1))))

            if logger.isEnabledFor(logging.DEBUG):
                self._feature_padding_length = _get_length()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._feature_padding_length = _get_length()

        return self._feature_padding_length

    @staticmethod
    def add_innermost_dimension(array: ak.Array) -> ak.Array:
        """Potentially add an inner dimension if the array is not 2D.

        Works by trying to access the first element at axis 1. If that element
        does not exist, add an inner dimension at axis 0 using ak.singletons().

        Args:
            array: ak.Array: The input data. If of shape (E, 1), nothing happens
                and the original array is returned. If of shape (E,), an inner
                dimension is added and the array is reshaped to (E, 1).

        Returns:
            ak.Array: The same array now shaped 2D (E, 1).

        Example:
            Case where the input array is 1D and needs an inner dimension added:
            >>> print(PaddedEagerDataset.add_innermost_dimension([11, 52, 31]))
            ak.Array([[11], [52], [31]])

            Case where the input array is already 2D and nothing needs to be done:
            >>> print(PaddedEagerDataset.add_innermost_dimension([[11], [52], [31]]))
            ak.Array([[11], [52], [31]])
        """
        try:
            _ = array[0][0]
        except IndexError:
            logger.debug("Added an extra dimension to the array at depth 0.")
            array = ak.singletons(array, axis=0)
        finally:
            return array
