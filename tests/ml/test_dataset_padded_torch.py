import pytest
from torch.utils.data import DataLoader

from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets import PaddedTorchDataset


@pytest.mark.skip("Not implemented yet")
@pytest.mark.parametrize("shuffle_partitions", [False, True])
@pytest.mark.parametrize("shuffle_events", [False, True])
def test_shuffling(ingestor: Ingestor, shuffle_partitions, shuffle_events):
    dataset = PaddedTorchDataset(
        features=ingestor,
        labels=ingestor,
        shuffle_partitions=shuffle_partitions,
        shuffle_events=shuffle_events,
        random_seed=42,
    )

    _ = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    # TODO Missing check shuffling
