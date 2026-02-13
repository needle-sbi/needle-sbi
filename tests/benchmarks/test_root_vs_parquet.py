"""
Compare the speed up between root and parquet files on the same Delphes dataset. Requires that
the Environment variable for both the root and parquet version of the datasets are defined. Will
convert the root files to parquet if not already done so.

Disclaimer: Part of this code was written with the help of GPT-5

Run these tests using the following command:

```python3
pytest --benchmark-only -s
```

The pytest mark `benchmark` is automatically added with the pytest fixture of the same name. This
test suite requires the specific Delphes dataset from KIT. There are two environment variables to
set:

```
export DELPHES_DATA_ROOT=/path/to/*.root
export DELPHES_DATA_PARQUET=/path/to/*.parquet
```

These must be a glob pattern of all the files. The columns and other configs are read from the
dedicated test `hydra_test_conf` directory for all tests. In that config it is not mandatory to set
the paths to the datasets because they are overwritten by the two environment variables mentioned
above.
"""

from pathlib import Path
from typing import Annotated, Callable, List, Literal

import pytest
from dask.distributed import Client
from pydantic import Field
from pytest_benchmark.fixture import BenchmarkFixture
from torch.utils.data import DataLoader

from orchestrator.config import MainConfig

pytest.importorskip("preprocessor", reason="Could not import 'preprocessor'")
from preprocessor.ingestion.formatter import Ingestor  # noqa: E402
from preprocessor.utils.array import resolve_paths  # noqa: E402
from preprocessor.utils.conversion import convert_root_to_parquet  # noqa: E402

pytest.importorskip("ml", reason="Could not import 'ml'")
from ml.data.padded.eager import PaddedDataset  # noqa: E402

Percentage = Annotated[float, Field(ge=0.0, le=1.0)]


class BenchmarkUtility:
    COLUMN_MODES = {
        pytest.param("one", id="columns=1"),
        pytest.param("config", id="columns=config"),
    }
    FILE_PERCENTAGE = [
        pytest.param(0.0, id="files=0%"),
        pytest.param(10.0, id="files=10%", marks=pytest.mark.slow),
        pytest.param(50.0, id="files=50%", marks=pytest.mark.slow),
        pytest.param(100.0, id="files=100%", marks=pytest.mark.slow),
    ]
    NUM_EVENTS = [
        pytest.param(10**3, id="events=1k"),
        pytest.param(10**5, id="events=100k", marks=pytest.mark.slow),
        pytest.param(10**7, id="events=10M", marks=pytest.mark.slow),
    ]

    @staticmethod
    def get_column(
        column_mode: str,
        columns: List[str] | None,
        drop_branches: List[str] = ["fBits"],
    ) -> List[str] | None:
        """BUG `drop_branches` will not be applied if `columns == None`"""
        if columns:
            columns = [col for col in columns if all(drop not in col for drop in drop_branches)]
        match column_mode:
            case "one":
                return [columns[0]] if columns else None
            case "config":
                return columns
            case "all" | None:
                return None

    @staticmethod
    def get_files(file_percentage: Percentage, paths: List[str]) -> List[str]:
        return paths[: max(1, int(len(paths) * file_percentage))]


def run_test(
    method: Literal["only_metadata", "materialize_partitions", "iterate_dataloader"],
    config: MainConfig,
    paths: List[str],
    drop_branches: List[str],
) -> Callable:
    """_summary_

    Args:
        method (Literal[&quot;only_metadata&quot;, &quot;materialize_partitions&quot;, &quot;iterate_dataloader&quot;]):
            Which kind of test to run from the list of implemented functions.
        config (MainConfig):
        paths (List[str]): List of paths to the data files. Valid paths are all paths accepted by `Ingestor`
        drop_branches (List[str]): List of potentially corrupted branches to drop at runtime

    Returns:
        Callable: A function without args that will run the desired test
    """

    def filter_name_func(name: str) -> bool:
        """Check if the str is in the list of branches to drop"""
        return not any(d in name for d in drop_branches)

    def _test_only_metadata():
        """Test function to read the metadata from the files

        Does not materialize partitions and does not perform any computation of the arrays.
        """
        _ = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.datasets.features_columns,
            max_number_events=config.datasets.max_number_events,
        )

    def _test_materialize_partitions():
        """Test function to materialize partitions from ingested data.

        Creates an Ingestor instance with specified configuration, filters columns
        based on a filter function, and computes the mapped partitions to materialize
        them in memory. Performs no actual calculation.
        """

        ingestor = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.datasets.features_columns,
            max_number_events=config.datasets.max_number_events,
        )
        arr = ingestor.array[[c for c in ingestor.fields if filter_name_func(c)]]
        arr.map_partitions(lambda x: x).compute()

    def _test_iterate_dataloader():
        """Test function to iterate through a dataloader with padded dataset.

        This function creates two Ingestor instances for features and labels,
        filters their columns based on the filter function, combines them into
        a PaddedDataset, and then iterates through a DataLoader to verify
        that the data pipeline works correctly without errors.

        The test verifies that:
        - Data can be loaded and filtered properly
        - The PaddedDataset correctly combines features and labels
        - The DataLoader can iterate through the dataset without exceptions
        """

        ingestor_features = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.datasets.features_columns,
            max_number_events=config.datasets.max_number_events,
        )
        ingestor_features.array = ingestor_features.array[[c for c in ingestor_features.fields if filter_name_func(c)]]
        ingestor_labels = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.datasets.features_columns,
            max_number_events=config.datasets.max_number_events,
        )
        ingestor_labels.array = ingestor_labels.array[[c for c in ingestor_labels.fields if filter_name_func(c)]]
        datamodule = PaddedDataset(
            ingestor_features,
            ingestor_labels,
        )
        dataloader = DataLoader(datamodule)
        for _ in dataloader:
            pass

    test_methods = {
        "only_metadata": _test_only_metadata,
        "materialize_partitions": _test_materialize_partitions,
        "iterate_dataloader": _test_iterate_dataloader,
    }

    return test_methods[method]


@pytest.mark.parametrize("file_percentage", BenchmarkUtility.FILE_PERCENTAGE)
@pytest.mark.parametrize("column_mode", BenchmarkUtility.COLUMN_MODES)
@pytest.mark.parametrize("num_events", BenchmarkUtility.NUM_EVENTS)
@pytest.mark.parametrize("file_type", ["root", "parquet"])
@pytest.mark.parametrize("test_method", ["only_metadata", "materialize_partitions", "iterate_dataloader"])
def test_ingestion_speed(
    benchmark: BenchmarkFixture,
    delphes_sample_root: str,
    delphes_sample_parquet: str,
    config_factory,
    dask_client: Client,
    column_mode: str,
    file_percentage: Percentage,
    file_type: str,
    test_method: Literal["only_metadata", "materialize_partitions", "iterate_dataloader"],
    num_events: int,
    drop_branches=["ref", "fName", "fSize", "fP", "fE", "fBits"],
) -> None:
    """_summary_

    Args:
        benchmark (BenchmarkFixture): Registers this test as a pytest-benchmark instance
        delphes_sample_root (str): Path to the Delphes (Root) samples
        delphes_sample_parquet (str): Path to the Delphes (Parquet)  samples. if empty, these files
            will be generated by converting the root files from `delphes_sample_root` to .parquet.
        config_factory (_type_): Load the config (as a factory to override some of the arguments)
        dask_client (Client): Run this code within the dask Client environment
        column_mode (str): Which columns to choose. See `BenchmarkUtility`
        file_percentage (Percentage): How many files to open. See `BenchmarkUtility`
        file_type (str): Run this test either as Root or Parquet files.
        num_events (int): How many events to load. Will cap at the maximal amount of events found in
            the loaded files.
        drop_branches (list, optional): Remove these branches when reading and converting files.
            Defaults to ["ref", "fName", "fSize", "fP", "fE", "fBits"], which are invalid branches
            in the default Delphes dataset.
    """
    if file_type == "parquet":
        convert_root_to_parquet(
            delphes_sample_root,
            Path(delphes_sample_parquet).parent,
            drop_branches=drop_branches,
        )
    config: MainConfig = config_factory(overrides=["datasets=delphes"])
    config.datasets.max_number_events = num_events
    config.datasets.features_columns = BenchmarkUtility.get_column(
        column_mode=column_mode,
        columns=config.datasets.features_columns,
        drop_branches=drop_branches,
    )
    paths_glob = delphes_sample_root or config.datasets.paths
    paths = BenchmarkUtility.get_files(
        file_percentage=file_percentage,
        paths=resolve_paths(paths_glob),
    )
    benchmark(
        run_test(method=test_method, config=config, paths=paths, drop_branches=drop_branches),
    )
