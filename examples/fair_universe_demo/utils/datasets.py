"""
Original authors: FAIR Universe HiggsML Challenge
Based on https://github.com/FAIR-Universe/HEP-Challenge
Adapted by K. Schmidt
"""

import json
import logging
import os
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

    def __init__(
        self,
        input_dir: str,
        parquet_filename: str = "FAIR_Universe_HiggsML_data.parquet",
        metadata_filename: str = "FAIR_Universe_HiggsML_data_metadata.json",
        test_size: Percentage = 0.3,
    ):
        """
        Constructs a Data object.

        Parameters:
            input_dir (str): The directory path of the input data.
        """
        train_data_file = os.path.join(input_dir, parquet_filename)
        croissant_file = os.path.join(input_dir, metadata_filename)

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

        self.parquet_file = pq.ParquetFile(train_data_file)

        # Step 1: Determine the total number of rows
        if "total_rows" in self.metadata:
            self.total_rows = self.metadata["total_rows"]
        else:
            # If total_rows is not in metadata, calculate it from the row groups
            self.total_rows = sum(
                self.parquet_file.metadata.row_group(i).num_rows for i in range(self.parquet_file.num_row_groups)
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
        self.__train_set = self.__load_data(selected_train_indices)

        # Balancing the weights

    def __load_data(self, selected_indices) -> pd.DataFrame:
        """Load selected rows from the parquet file into a pandas DataFrame.

        Args:
            selected_indices (np.ndarray): Sorted row indices to read.

        Returns:
            pd.DataFrame: DataFrame containing the selected rows.
        """
        current_row = 0
        sampled_df = pd.DataFrame()

        chunks = []
        for row_group_index in tqdm(
            range(self.parquet_file.num_row_groups),
            total=self.parquet_file.num_row_groups,
            unit="row_groups",
            desc="Loading data from parquet file",
        ):
            row_group = self.parquet_file.read_row_group(row_group_index).to_pandas()
            row_group_size = len(row_group)
            within_group_indices = (
                selected_indices[(selected_indices >= current_row) & (selected_indices < current_row + row_group_size)]
                - current_row
            )
            if len(within_group_indices) > 0:
                chunks.append(row_group.iloc[within_group_indices])
            current_row += row_group_size

        sampled_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

        if "sum_weights" in self.metadata:
            sum_weights = self.metadata["sum_weights"]
            if sum_weights > 0:
                sampled_df["weights"] = (sum_weights * sampled_df["weights"]) / sum(sampled_df["weights"])
            else:
                logger.warning("Sum of weights is zero. No balancing applied.")

        return sampled_df

    def load_test_set(self):
        """Load the test dataset from the parquet file.

        Side effects:
            Sets `self.__test_set` with labeled subsets.
        """
        selected_test_indices = np.array(range(self.test_size))
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
