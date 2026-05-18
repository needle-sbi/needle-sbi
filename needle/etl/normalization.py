"""
Compute normalization constants from Parquet file metadata.

Collects per-file (partition) column statistics such as min/max,
without reading any data into memory.
"""
from __future__ import annotations

from typing import Any, Protocol

import awkward as ak
import dask_awkward as dak
import pyarrow.parquet as pq

from needle.etl.array import NestedArrayIndexer
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

#: Nested metadata structure: ``{field → {scaler_key → {class_index → value}}}``.
type MetaDataDict = dict[str, dict[str, dict[int, Any]]]


def clean_column_name(path: str) -> str:
    """
    Clean up pyarrow column names to the one actually in the parquet file

    Important:
        Will filter out the fields "list" and "item".

    Example:
        >>> clean_column_name('Lepton.list.item.eta')
        'Lepton.eta'

        >>> clean_column_name('event')
        'event'
    """
    parts = path.split(".")
    filtered = [p for p in parts if p not in {"list", "item"}]
    return ".".join(filtered)


def gather_metadata(paths: list[str]) -> MetaDataDict:
    """
    Collect min/max stats for all columns in each Parquet file using pyarrow.

    Args:
        paths: list of Parquet file paths.

    Returns:
        dict:
            A nest dict of shape:
                {<column_name>: {"min": {<partition_id>: <val>}, "max": ...}, ...}, ...}
    """
    metadata: MetaDataDict = {}

    for pid, path in enumerate(paths):
        file_meta = pq.read_metadata(path)

        n_row_groups = file_meta.num_row_groups

        per_file_stats: dict[str, dict[str, Any]] = {}

        for row_group_index in range(n_row_groups):
            row_group = file_meta.row_group(row_group_index)

            for col_idx in range(row_group.num_columns):
                column = row_group.column(col_idx)
                name = clean_column_name(str(column.path_in_schema))
                stats = column.statistics

                if (stats is None) or (stats.min is None) or (stats.max is None):
                    continue

                cmin = stats.min if not hasattr(stats.min, "as_py") else stats.min.as_py()
                cmax = stats.max if not hasattr(stats.max, "as_py") else stats.max.as_py()

                if name not in per_file_stats:
                    per_file_stats[name] = {"min": cmin, "max": cmax}
                else:
                    per_file_stats[name]["min"] = min(per_file_stats[name]["min"], cmin)
                    per_file_stats[name]["max"] = max(per_file_stats[name]["max"], cmax)

        for col, stats in per_file_stats.items():
            col_meta = metadata.setdefault(col, {"min": {}, "max": {}})
            col_meta["min"][pid] = stats["min"]
            col_meta["max"][pid] = stats["max"]

    return metadata


class ScalerProtocol(Protocol):
    cache: dict[str, dict[str, Any]]
    """Format: `{"<metric>": {"<field>": <value>}, ...}`"""

    def apply(self, array: dak.Array) -> dak.Array:
        ...

    def revert(self, array: dak.Array) -> dak.Array:
        ...


class MinMaxScaler(ScalerProtocol):
    def apply(self, array: dak.Array) -> dak.Array:
        """Apply a min-max transformation unto a delayed array

        Transformation:
            (x - min) / (max - min)

        Where max, min are calculated independently for each field over the whole array.

        Args:
            array (dask_awkward.Array): The Array to normalize

        Returns:
            dask_awkward.Array

        Note:
            The way min and max are calculated are not optimal since we load in the full arrays each
            time for each field. On the other hand, using `dak.min` fails due to a known bug.
        """
        self.cache = {"min": dict(), "max": dict()}
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")

        for field in fields:
            minimum = self.cache["min"].get(field)
            maximum = self.cache["max"].get(field)

            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")

            if minimum is None:
                minimum = ak.min(column.compute())
                self.cache["min"][field] = minimum

            if maximum is None:
                maximum = ak.max(column.compute())
                self.cache["max"][field] = maximum

            column = (column - minimum) / (maximum - minimum)

            self.cache["min"][field] = minimum
            self.cache["max"][field] = maximum

        return array

    def revert(self, array: dak.Array) -> dak.Array:
        """Undo the normalization transformation

        Transformation:
            x * (max - min) + min

        Args:
            array (dask_awkward.Array): The Array to un-normalize

        Returns:
            dask_awkward.Array
        """
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")

        for field in fields:
            minimum = self.cache["min"][field]
            maximum = self.cache["max"][field]

            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")
            column = column * (maximum - minimum) + minimum

        return array


class StandardScaler(ScalerProtocol):
    def apply(self, array: dak.Array) -> dak.Array:
        """Apply a standard scaling transformation unto a delayed array

        Transformation:
            (x - mean) / std

        Where mean, std are calculated independently for each field over the whole array.

        Args:
            array (dask_awkward.Array): The Array to normalize

        Returns:
            dask_awkward.Array
        """

        self.cache = {"mean": dict(), "std": dict()}
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")

        for field in fields:
            mean = self.cache["mean"].get(field)
            std = self.cache["std"].get(field)

            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")

            if mean is None:
                mean = ak.mean(column.compute())
                self.cache["mean"][field] = mean

            if std is None:
                std = ak.std(column.compute())
                self.cache["std"][field] = std

            column = (column - mean) / std

        return array

    def revert(self, array: dak.Array) -> dak.Array:
        """Undo the normalization transformation

        Transformation:
            x * std + mean

        Args:
            array (dask_awkward.Array): The Array to un-normalize

        Returns:
            dask_awkward.Array
        """
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")

        for field in fields:
            mean = self.cache["mean"][field]
            std = self.cache["std"][field]

            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")
            column = column * std + mean

        return array
