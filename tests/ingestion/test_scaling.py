import awkward as ak

from needle.etl.dask_ingestor import Ingestor
from needle.etl.normalization import MinMaxScaler, StandardScaler


def test_minmax_scaler(
    ingestor: Ingestor,
):
    field = "Lepton.pt"
    min = ak.min(ingestor[field].compute())
    max = ak.max(ingestor[field].compute())

    scaler = MinMaxScaler()
    ingestor.array = scaler.revert(scaler.apply(ingestor.array))

    assert min == ak.min(ingestor[field].compute())
    assert max == ak.max(ingestor[field].compute())


def test_standard_scaler(
    ingestor: Ingestor,
):
    field = "Lepton.pt"
    mean = ak.mean(ingestor[field].compute())
    std = ak.std(ingestor[field].compute())

    scaler = StandardScaler()
    ingestor.array = scaler.revert(scaler.apply(ingestor.array))

    assert mean == ak.mean(ingestor[field].compute())
    assert std == ak.std(ingestor[field].compute())
