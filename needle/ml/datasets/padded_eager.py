"""
Base class for Datasets that handle ragged data from dask_awkward arrays. This class
will load all the events into memory at instantiation, padding them to the maximum number
of particles in the feature array.
"""

import logging
import warnings

import awkward as ak
import torch

from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets.padded_base import PaddedDatasetBase, logger


class PaddedDataset(PaddedDatasetBase):
    SHUFFLE_ALLOWED: bool = True
    TORCH_MULTIPROCESSING_ALLOWED: bool = True

    def __init__(
        self,
        features: Ingestor,
        labels: Ingestor,
    ):
        """
        Dataset structured as (E,P,F) tensors, where E is the number of events,
        P is the number of particles per event, and F is the number of features per particle.
        The dataset is padded to the maximum number of particles per event and computed
        eagerly at instantiation.

        Once loaded with torch.utils.data.DataLoader, the dataset can be accessed
        with the usual way:
            dataset[i] -> returns a tuple (x,y) of tensors of shape (P,F) for event i

        Args:
            features (dak.Array): Dask_awkward Record array containing feature columns,
                where each field is a feature name and the data has shape (E, P).
            labels (dak.Array): Dask_awkward Record array containing label columns,
                where each field is a label name and the data has shape (E, P).

        Example:

            dataset = PaddedEagerDataset(
                features=dak.array[['feature1', 'feature2', 'feature3']],
                labels=dak.array[['label1', 'label2']]
            )

            Make sure all columns have the same shape (E, P) and can be concatenated along
            a new axis.

        Methods:
            __getitem__(idx): Returns a tuple (x, y) of tensors of shape (P, F) for event idx,
                where x contains the features and y contains the labels.
            __len__(): Returns the number of events in the dataset.

        Attributes:
            features_array (dak.Array): Dask_awkward Record array containing feature columns.
            labels_array (dak.Array): Dask_awkward Record array containing label columns.
            feature_names (list): List of feature names extracted from features_array.fields.
            labels_names (list): List of label names extracted from labels_array.fields.
            feature_tensors (torch.Tensor): Precomputed feature tensors of shape (E, P, F).
            label_tensors (torch.Tensor): Precomputed label tensors of shape (E, P, F).
            feature_padding_length (int): Maximum number of particles per event, used for padding. Computed
                once at instantiation and cached.

        Class attributes:
            SHUFFLE_ALLOWED (bool): Whether the dataset can be shuffled by the DataLoader.
                Default is True (normal pytorch behavior), but can be overridden by daughter classes.
            TORCH_MULTIPROCESSING_ALLOWED (bool): Whether the dataset can be used with
                torch multiprocessing. Default is True. Daughter classes that use dask or other
                multiprocessing libraries should consider setting this Flag to False.

        NOTE: This class loads all data into memory at instantiation for fast access during training.
        If this behavior is not desired, inherit from this class and overwrite the __init__ method.
        Some methods are only valid for ak.Arrays (so after computing the dask graph).
        """
        self.features_ingestor = features
        self.labels_ingestor = labels

        self.feature_names = features.fields
        self.labels_names = labels.fields

        self.feature_tensors = self.convert_ak_to_tensor(
            self.features_ingestor.array.compute(),
            self.feature_names,
        )
        self.label_tensors = self.convert_ak_to_tensor(
            self.labels_ingestor.array.compute(),
            self.labels_names,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_tensors[idx]
        y = self.label_tensors[idx]
        return (x, y)

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

            def _get_length():
                return ak.max(ak.ravel(ak.num(array, axis=1)))

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
            logger.warning("Added an extra dimension to the array at depth 0.")
            array = ak.singletons(array, axis=0)
        finally:
            return array
