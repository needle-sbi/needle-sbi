"""
Original authors: FAIR Universe HiggsML Challenge
Based on https://github.com/FAIR-Universe/HEP-Challenge
Adapted by K. Schmidt
"""

import json
import logging
import os
from pathlib import Path
from shutil import copy2
from typing import Annotated, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import Field
from tqdm import tqdm

from .systematics import systematics

Percentage = Annotated[float, Field(ge=0.0, le=1.0)]


logger = logging.getLogger("FAIR-Universe-Data")
ZENODO_URL = "https://zenodo.org/records/15131565/files/FAIR_Universe_HiggsML_data.zip?download=1"
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
THIS_FILE_PARENT_DIR = os.path.dirname(THIS_FILE_DIR)


class Data:
    """
    A class to represent a dataset.

    Parameters:
        * input_dir (str): The directory path of the input data.

    Attributes:
        * __train_set (dict): A dictionary containing the train dataset.
        * __test_set (dict): A dictionary containing the test dataset.
        * input_dir (str): The directory path of the input data.

    Methods:
        * load_train_set(): Loads the train dataset.
        * load_test_set(): Loads the test dataset.
        * get_train_set(): Returns the train dataset.
        * get_test_set(): Returns the test dataset.
        * delete_train_set(): Deletes the train dataset.
        * get_syst_train_set(): Returns the train dataset with systematic variations.
    """

    __train_set: pd.DataFrame
    __test_set: Dict[str, pd.DataFrame]
    cache_parquet_file: Path

    def __init__(
        self,
        input_dir: str,
        parquet_filename: str = "FAIR_Universe_HiggsML_data.parquet",
        metadata_filename: str = "FAIR_Universe_HiggsML_data_metadata.json",
        cache_parquet_file: str | None = None,
        test_size: Percentage = 0.3,
    ):
        """
        Constructs a Data object.

        Parameters:
            input_dir (str): The directory path of the input data.
        """
        input_path = Path(input_dir).expanduser()
        if input_path.is_file():
            data_file = input_path
        else:
            candidate_data_files = [
                input_path / parquet_filename,
                input_path / "data.parquet",
                input_path / "input_data" / "train" / "data" / "data.parquet",
            ]
            data_file = next((path for path in candidate_data_files if path.exists()), candidate_data_files[0])

        candidate_metadata_files = [
            data_file.with_name(metadata_filename),
            input_path / metadata_filename,
        ]
        metadata_file = next((path for path in candidate_metadata_files if path.exists()), candidate_metadata_files[0])

        self.train_data_file = str(data_file)
        croissant_file = str(metadata_file)
        split_train_dir = data_file.parent.parent
        self.split_sidecar_dir = (
            split_train_dir
            if (split_train_dir / "detailed_labels" / "data.detailed_labels").exists()
            and (split_train_dir / "labels" / "data.labels").exists()
            and (split_train_dir / "weights" / "data.weights").exists()
            else None
        )

        try:
            with open(croissant_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            logger.warning("Metadata file not found. Proceeding without metadata.")
            self.metadata = {}
        except json.JSONDecodeError:
            logger.warning("Metadata file is not a valid JSON. Proceeding without metadata.")
            self.metadata = {}
        except Exception as e:
            logger.warning(f"An error occurred while reading the metadata file: {e}")
            self.metadata = {}

        # Cache file on fastest available storage
        if cache_parquet_file:
            self.cache_parquet_file = Path(cache_parquet_file)
            os.makedirs(self.cache_parquet_file.parent, exist_ok=True)

            if not self.cache_parquet_file.exists():
                logger.info(
                    f"Caching input file '{self.train_data_file}' to '{self.cache_parquet_file} for faster access. "
                    "Disable by setting `Data(cache_parquet_file=...)` to None or False"
                )
                copy2(self.train_data_file, self.cache_parquet_file)

            self.train_data_file = self.cache_parquet_file

        logger.info(f"Opening file {self.train_data_file}")
        parquet_file = pq.ParquetFile(self.train_data_file)

        # Step 1: Determine the total number of rows
        if "total_rows" in self.metadata:
            self.total_rows = self.metadata["total_rows"]
        else:
            # If total_rows is not in metadata, calculate it from the row groups
            self.total_rows = sum(
                parquet_file.metadata.row_group(i).num_rows for i in range(parquet_file.num_row_groups)
            )

        if test_size is not None:
            if isinstance(test_size, int):
                test_size = min(test_size, self.total_rows)
            elif isinstance(test_size, float):
                if 0.0 <= test_size <= 1.0:
                    test_size = int(test_size * self.total_rows)
                else:
                    raise ValueError("Test size must be between 0.0 and 1.0")
            else:
                raise ValueError("Test size must be an integer or a float")

        self.test_size = test_size

    def print_dataset_info(self):
        logger.info(f"Total number of events: {self.total_rows}")
        logger.info(f"Number of training samples: {self.train_size}")
        logger.info(f"Number of testing samples: {self.test_size}")

    def load_train_set(self, train_size: int = None, selected_indices: List[int] | np.ndarray = None):
        """Load the training subset from the parquet dataset.

        Args:
            train_size (int | float | None): Number of rows or fraction of rows to load.
            selected_indices (list | np.ndarray | None): Specific row indices to include.

        Raises:
            ValueError: If sample size or selected indices are invalid.

        Side effects:
            Sets `self.__train_set`.
        """
        if train_size is not None:
            if isinstance(train_size, int):
                train_size = min(train_size, self.total_rows - self.test_size)
            elif isinstance(train_size, float):
                if 0.0 <= train_size <= 1.0:
                    train_size = int(train_size * (self.total_rows - self.test_size))
                else:
                    raise ValueError("Sample size must be between 0.0 and 1.0")
            else:
                raise ValueError("Sample size must be an integer or a float")

        elif selected_indices is not None:
            if isinstance(selected_indices, list):
                selected_indices = np.array(selected_indices)
            elif isinstance(selected_indices, np.ndarray):
                pass
            else:
                raise ValueError("Selected indices must be a list or a numpy array")
            train_size = len(selected_indices)
        else:
            train_size = self.total_rows - self.test_size

        if train_size > self.total_rows - self.test_size:  # type: ignore
            raise ValueError("Sample size exceeds the number of available rows")

        if selected_indices is None:
            selected_indices = np.random.choice(
                (self.total_rows - self.test_size),
                size=train_size,
                replace=False,
            )  # type: ignore

        selected_train_indices = np.sort(selected_indices) + self.test_size  # type: ignore
        self.train_size = len(selected_train_indices)
        self.__train_set = self.__load_data(selected_train_indices)

        # Balancing the weights

    def __load_data(self, selected_indices) -> pd.DataFrame:
        """Load selected rows from the parquet file into a pandas DataFrame.

        Args:
            selected_indices (np.ndarray): Sorted row indices to read.

        Returns:
            pd.DataFrame: DataFrame containing the selected rows.
        """
        parquet_file = pq.ParquetFile(self.train_data_file)
        current_row = 0
        sampled_df = pd.DataFrame()
        max_selected_index = int(selected_indices[-1]) if len(selected_indices) else -1

        chunks = []
        for row_group_index in tqdm(
            range(parquet_file.num_row_groups),
            total=parquet_file.num_row_groups,
            unit="row_groups",
            desc="Loading data from parquet file",
        ):
            row_group = parquet_file.read_row_group(row_group_index).to_pandas()
            row_group_size = len(row_group)
            within_group_indices = (
                selected_indices[(selected_indices >= current_row) & (selected_indices < current_row + row_group_size)]
                - current_row
            )
            if len(within_group_indices) > 0:
                chunks.append(row_group.iloc[within_group_indices])
            current_row += row_group_size
            if current_row > max_selected_index:
                break

        sampled_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

        if self.split_sidecar_dir and "detailed_labels" not in sampled_df.columns:
            sampled_df["detailed_labels"] = self.__load_sidecar_values(
                self.split_sidecar_dir / "detailed_labels" / "data.detailed_labels",
                selected_indices,
            )
            sampled_df["labels"] = self.__load_sidecar_values(
                self.split_sidecar_dir / "labels" / "data.labels",
                selected_indices,
                dtype="float32",
            )
            sampled_df["weights"] = self.__load_sidecar_values(
                self.split_sidecar_dir / "weights" / "data.weights",
                selected_indices,
                dtype="float32",
            )

        if "sum_weights" in self.metadata:
            sum_weights = self.metadata["sum_weights"]
            if sum_weights > 0:
                sampled_df["weights"] = (sum_weights * sampled_df["weights"]) / sum(sampled_df["weights"])
            else:
                logger.warning("Sum of weights is zero. No balancing applied.")

        return sampled_df

    @staticmethod
    def __load_sidecar_values(path: Path, selected_indices, dtype: str | None = None) -> pd.Series:
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        selected_set = set(selected_indices.tolist())
        max_index = int(selected_indices[-1]) if len(selected_indices) else -1

        values = []
        with open(path, "r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index in selected_set:
                    values.append(line.strip())
                if index >= max_index:
                    break

        return pd.Series(values, dtype=dtype)

    def load_test_set(
        self,
        test_size: int | None = None,
        selected_indices: List[int] | np.ndarray | None = None,
        random_seed: int | None = None,
        max_source_rows: int | None = None,
    ):
        """Load the test dataset from the parquet file.

        Args:
            test_size: Number of rows to load. Defaults to ``self.test_size``.
            selected_indices: Explicit row indices to use for the test set.
            random_seed: Seed for drawing a representative subset when ``selected_indices`` is not supplied.
            max_source_rows: Restrict random sampling to the first N rows. Useful for local debugging on large split
                datasets because sidecar label files are line-oriented.

        Side effects:
            Sets `self.__test_set` with labeled subsets.
        """
        if selected_indices is not None:
            selected_test_indices = np.sort(np.asarray(selected_indices, dtype=np.int64))
        else:
            if test_size is None:
                test_size = self.test_size
            if test_size is None:
                test_size = self.total_rows

            source_rows = self.total_rows if max_source_rows is None else min(max_source_rows, self.total_rows)
            test_size = min(test_size, source_rows)

            if random_seed is None:
                selected_test_indices = np.arange(test_size)
            else:
                random_state = np.random.default_rng(random_seed)
                selected_test_indices = np.sort(random_state.choice(source_rows, size=test_size, replace=False))

        test_df = self.__load_data(selected_test_indices)

        keys = ["ztautau", "diboson", "ttbar", "htautau"]
        test_set = {}

        for key, group in test_df[test_df["detailed_labels"].isin(keys)].groupby("detailed_labels"):
            df = group.copy()
            df.loc[:, "Label"] = df["detailed_labels"]
            test_set[key] = df

        for key in keys:
            test_set.setdefault(key, pd.DataFrame())

        self.__test_set = test_set

    def get_train_set(self):
        """Return the loaded training dataset.

        Returns:
            pd.DataFrame: The training dataset loaded by `load_train_set`.
        """
        train_set = self.__train_set
        return train_set

    def get_test_set(self):
        """Return the loaded test dataset.

        Returns:
            dict: Dictionary of labeled test subsets.
        """
        return self.__test_set

    def delete_train_set(self):
        """Delete the cached training dataset from memory.

        Side effects:
            Removes `self.__train_set`.
        """
        del self.__train_set

    def get_syst_train_set(
        self,
        tes=1.0,
        jes=1.0,
        soft_met=0.0,
        ttbar_scale=None,
        diboson_scale=None,
        bkg_scale=None,
        dopostprocess=False,
    ):
        """Return training data with systematic variations applied.

        Args:
            tes (float): Tau energy scale variation.
            jes (float): Jet energy scale variation.
            soft_met (float): Soft MET variation.
            ttbar_scale (float | None): ttbar background normalization factor.
            diboson_scale (float | None): Diboson background normalization factor.
            bkg_scale (float | None): Background normalization factor.
            dopostprocess (bool): Whether to apply postprocessing.

        Returns:
            dict: Systematically varied training data.
        """
        if self.__train_set is None:
            self.load_train_set()

        return systematics(
            data_set=self.__train_set,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
            bkg_scale=bkg_scale,
            dopostprocess=dopostprocess,
        )
