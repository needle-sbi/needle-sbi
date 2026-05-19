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
dedicated test `conf_tests` directory for all tests. In that config it is not mandatory to set
the paths to the datasets because they are overwritten by the two environment variables mentioned
above.
"""

from pathlib import Path
from typing import Annotated, Callable, Dict, List, Literal

import pydantic
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from torch.utils.data import DataLoader

from needle.etl.array import resolve_paths
from needle.etl.conversion import convert_root_to_parquet
from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets import PaddedDataset
from needle.utils.config_schema import DatasetConfig, EstimatorConfig, ExpansionConfig

Percentage = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]


@pytest.fixture()
def benchmark_config() -> EstimatorConfig:
    """Fixture that provides a standalone EstimatorConfig instance for benchmark tests.

    Creates a minimal EstimatorConfig with a default estimator configuration.
    No external config files or hydra loading is used.

    Returns:
        EstimatorConfig
    """
    # Create a default dataset configuration for ingestion tests
    dataset_config = DatasetConfig(
        paths="",  # Will be overridden by the test
        features_columns=[],  # Will be set by BenchmarkUtility.get_column()
        labels_columns=[],
        format="automatic",
        max_number_events=-1,  # Will be set by the test
    )

    # Create a default estimator configuration
    estimator_config = EstimatorConfig(
        datamodule="default",
        datamodule_override=None,
        dataset="default",
        dataset_override=dataset_config,
        model="default",
        model_override=None,
        trainer="default",
        trainer_override=None,
        expands=ExpansionConfig(),
        requires=None,
    )

    return estimator_config


class BenchmarkUtility:
    COLUMN_MODES = {
        pytest.param("one", id="columns_1"),
        pytest.param("config", id="columns_config"),
    }
    FILE_PERCENTAGE = [
        pytest.param(0.0, id="files_0percent"),
        pytest.param(10.0, id="files_10percent", marks=pytest.mark.slow),
        pytest.param(50.0, id="files_50percent", marks=pytest.mark.slow),
        pytest.param(100.0, id="files_100percent", marks=pytest.mark.slow),
    ]
    NUM_EVENTS = [
        pytest.param(10**3, id="events_1k"),
        pytest.param(10**5, id="events_100k", marks=pytest.mark.slow),
        pytest.param(10**7, id="events_10M", marks=pytest.mark.slow),
        pytest.param(-1, id="events_all", marks=pytest.mark.slow),
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
    @pydantic.validate_call
    def get_files(file_percentage: Percentage, paths: List[str]) -> List[str]:
        return paths[: max(1, int(len(paths) * file_percentage))]


def run_test(
    method: Literal["only_metadata", "materialize_partitions", "iterate_dataloader"],
    config: EstimatorConfig,
    paths: List[str],
    drop_branches: List[str],
    file_type: Literal["parquet", "root"],
) -> Callable:
    """Benchmark between root and parquet file ingestion with dask_awkward

    Args:
        method (Literal[&quot;only_metadata&quot;, &quot;materialize_partitions&quot;, &quot;iterate_dataloader&quot;]):
            Which kind of test to run from the list of implemented functions.
        config (MainConfig):
        paths (List[str]): List of paths to the data files. Valid paths are all paths accepted by `Ingestor`
        drop_branches (List[str]): List of potentially corrupted branches to drop at runtime

    Returns:
        Callable: A function without args that will run the desired test
    """

    assert config.dataset_override is not None

    def filter_name_func(columns: List[str]) -> Callable[[str], bool]:
        """Check if the str is in the list of branches to drop"""

        def _filter(name: str) -> bool:
            is_valid = not any(d in name for d in drop_branches)
            is_in_columns = name in columns
            return is_valid and is_in_columns

        return _filter

    def reader_kwargs() -> Dict[str, Callable]:
        assert config.dataset_override is not None
        assert config.dataset_override.features_columns is not None
        match file_type:
            case "parquet":
                return {}
            case "root":
                return {"filter_name": filter_name_func(config.dataset_override.features_columns)}

    def _test_only_metadata():
        """Test function to read the metadata from the files

        Does not materialize partitions and does not perform any computation of the arrays.
        """
        assert config.dataset_override is not None
        _ = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.dataset_override.features_columns,
            max_number_events=config.dataset_override.max_number_events,
            reader_kwargs=reader_kwargs(),
        )

    def _test_materialize_partitions():
        """Test function to materialize partitions from ingested data.

        Creates an Ingestor instance with specified configuration, filters columns
        based on a filter function, and computes the mapped partitions to materialize
        them in memory. Performs no actual calculation.
        """
        assert config.dataset_override is not None
        ingestor = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.dataset_override.features_columns,
            max_number_events=config.dataset_override.max_number_events,
            reader_kwargs=reader_kwargs(),
        )
        for field in ingestor.fields:
            ingestor[field].compute()

    def _test_iterate_dataloader():
        """Test function to iterate through a dataloader with padded dataset.

        This function creates an Ingestor instance and filters the columns based on the filter
        function, combines them into a PaddedDataset, and then iterates through a DataLoader to
        verify that the data pipeline works correctly without errors.

        The test verifies that:
        - Data can be loaded and filtered properly
        - The DataLoader can iterate through the dataset without exceptions
        """
        assert config.dataset_override is not None
        ingestor = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.dataset_override.features_columns,
            max_number_events=config.dataset_override.max_number_events,
            reader_kwargs=reader_kwargs(),
        )
        datamodule = PaddedDataset(ingestor, ingestor)
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
    benchmark_config: EstimatorConfig,
    delphes_sample_root: str,
    delphes_sample_parquet: str,
    column_mode: str,
    file_percentage: Percentage,
    file_type: Literal["parquet", "root"],
    test_method: Literal["only_metadata", "materialize_partitions", "iterate_dataloader"],
    num_events: int,
    drop_branches=["ref", "fName", "fSize", "fP", "fE", "fBits"],
) -> None:
    """Test function to compare the ingestion speeds of parquet and root files in different scenarios.

    The larger and longer test for many events are marked as slow. To run them, add the '-m slow' marker
    when running the tests. More info is given in the corresponding `BenchmarkUtility` class.

    Args:
        benchmark (BenchmarkFixture): Registers this test as a pytest-benchmark instance
        benchmark_config (EstimatorConfig): Configuration instance for benchmark tests
        delphes_sample_root (str): Path to the Delphes (Root) samples
        delphes_sample_parquet (str): Path to the Delphes (Parquet) samples. if empty, these files
            will be generated by converting the root files from `delphes_sample_root` to .parquet.
        column_mode (str): Which columns to choose. See `BenchmarkUtility`
        file_percentage (Percentage): How many files to open. See `BenchmarkUtility`
        file_type (str): Run this test either as Root or Parquet files.
        test_method (str): Which benchmark test method to run. See `BenchmarkUtility`
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
        data_path = delphes_sample_parquet
    else:
        data_path = delphes_sample_root

    config: EstimatorConfig = benchmark_config
    # Access dataset config through the default estimator
    dataset_config = config.dataset_override
    assert dataset_config is not None
    dataset_config.max_number_events = num_events
    dataset_config.features_columns = BenchmarkUtility.get_column(
        column_mode=column_mode,
        columns=dataset_config.features_columns,
        drop_branches=drop_branches,
    )
    paths = BenchmarkUtility.get_files(
        file_percentage=file_percentage,
        paths=resolve_paths(data_path),
    )
    benchmark(
        run_test(
            method=test_method,
            config=config,
            paths=paths,
            drop_branches=drop_branches,
            file_type=file_type,
        ),
    )
