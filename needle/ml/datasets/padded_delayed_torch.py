import numpy as np
import torch
from torch.utils.data import IterableDataset

from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets.io import PartitionQueue
from needle.ml.datasets.kfold import KFold
from needle.ml.datasets.padded_base import PaddedDatasetBase


class PaddedTorchDataset(IterableDataset, PaddedDatasetBase):
    """Out-of-memory dataset using torch multiprocessing and dask single-thread read-in."""

    SHUFFLE_ALLOWED: bool = False
    TORCH_MULTIPROCESSING_ALLOWED: bool = True

    def __init__(
        self,
        features: Ingestor,
        labels: Ingestor,
        shuffle_partitions: bool = False,
        shuffle_events: bool = True,
        random_seed: int = 42,
        kfold: KFold | None = None,
    ):
        """This class extends the PaddedEagerDataset to support chunked iterating with Torch.

        In the parent class, the data is loaded during instantiation. Instead, this class focusses
        on combining the __iter__ method of IterableDataset with the PaddedEagerDataset's functionality to
        yield chunks of data. The __iter__ method still yields individual events, same as the regular
        __getitem__ method of the parent class, but loads only discrete chunks of data to memory.
        This is compatible with multiple workers if the Dataloader is using num_workers > 0.

        Args:
            features (dict[str, dak.Array]): Dictionary of dask_awkward arrays for features
            labels (dict[str, dak.Array]): Dictionary of dask_awkward arrays for labels
            chunk_size (int): Number of events per chunk to load into memory at once.
            shuffle_chunks (bool): Whether to shuffle the chunks before yielding.
            shuffle_events (bool): Whether to shuffle the events within each chunk.
            random_seed (int): Random seed for reproducibility.
            kfold (KFold): Instance of the KFold class

        Important:
            This dataset cannot be shuffled by the Dataloader, as the chunks are loaded in sequence,
            and shuffling it would invalidate the chunking. Instead, the shuffling is configured by
            the arguments `shuffle_chunks` and `shuffle_events`.
        """
        self.features_ingestor = features
        self.labels_ingestor = labels

        self.shuffle_partitions = shuffle_partitions
        self.shuffle_events = shuffle_events
        self.random_seed = random_seed
        self.kfold = kfold

        self.feature_names = features.fields
        self.labels_names = labels.fields

        self.features_queue = PartitionQueue(features.array)
        self.labels_queue = PartitionQueue(labels.array)

    def __iter__(self):
        """Yields individual events from the dataset in after loading them in chunks.

        If the Dataloader is using multiple workers, each worker will process a different set of
        chunks based on the worker ID (as determined by torch.utils.data.get_worker_info()).

        Yields:
            tuple: A tuple (features, labels) of torch Tensors

        Note:
            Optionally, both the chunks and the events can be shuffled before yielding.
            During regular training, it is recommended to set both to True.

            The splitting of the data works as follows:
            - The dataset is split into chunks of size `chunk_size`.
            - Each worker processes a different set of chunks based on its ID. The
                actual chunk index can be shuffled beforehand.
            - Within each chunk, the events are loaded into memory using the
                `_compute_tensors` method, which pads the features and labels to the
                maximum number of particles per event. Same as in the parent class,
                Afterwards, the events are optionally shuffled by permuting their
                indices. Finally, each event is yielded as a tuple of tensors
                (features, labels) of shape (P, F).
        """
        rng = np.random.default_rng(self.random_seed)
        self.worker_info = torch.utils.data.get_worker_info()

        if self.kfold:
            if self.worker_info:
                assigned_partition_mask = [
                    p % self.worker_info.num_workers == self.worker_info.id for p in self.kfold.partition_ids
                ]
                self.kfold.mask(assigned_partition_mask)
            if self.shuffle_partitions:
                self.kfold.shuffle(rng)
            partitions = self.kfold.partitions
        else:
            partition_ids = list(range(self.features_ingestor.array.npartitions))
            if self.worker_info:
                partition_ids = [p for p in partition_ids if p % self.worker_info.num_workers == self.worker_info.id]
            if self.shuffle_partitions:
                partition_ids = rng.permutation(partition_ids)
            partitions = {p: None for p in partition_ids}

        for partition_id, slicing_index in partitions.items():
            features_partition = self.features_queue.load_partition_thread_safe(partition_id, slicing_index)
            labels_partition = self.labels_queue.load_partition_thread_safe(partition_id, slicing_index)

            features_tensor = self.convert_ak_to_tensor(
                features_partition.compute(),
                self.feature_names,
            )
            labels_tensor = self.convert_ak_to_tensor(
                labels_partition.compute(),
                self.labels_names,
            )

            event_indices = range(len(features_tensor))

            if self.shuffle_events:
                event_indices = rng.permutation(event_indices)

            for event_idx in event_indices:
                yield (features_tensor[event_idx], labels_tensor[event_idx])
