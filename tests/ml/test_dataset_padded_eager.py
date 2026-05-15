import pytest
import torch

from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets import PaddedDataset


def test_init(ingestor: Ingestor):
    if not ingestor:
        pytest.skip("Ingestor could not be created, skipping test.")
    else:
        dataset = PaddedDataset(features=ingestor, labels=ingestor)
        item = dataset[0]
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], torch.Tensor)
        assert item[0].shape[0] == dataset._feature_padding_length
        assert item[1].shape[0] == dataset._feature_padding_length
