"""Utilities for handling files"""

import functools
import glob
from typing import Literal, overload

import awkward as ak
import dask_awkward as dak
import numpy as np
import pyarrow.parquet as pq
from awkward.errors import FieldNotFoundError
from pyarrow import ArrowInvalid

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("etl")


def resolve_paths(
    paths: str | list[str],
) -> list[str]:
    """Resolve a unique list of files based on glob

    The result can be used to loop over all files without having to consider glob patterns. All
    files will be listed individually in the output list.

    Args:
        paths: str | list[str]: Reference to files in a str form with potential wildcards
            or a list of str also with wildcards

    Returns:
        list[str]: A list of str as resolved using glob.
    """
    resolved: list[str] = []

    if isinstance(paths, str):
        resolved.extend(glob.glob(paths, recursive=True))
    else:
        for path in paths:
            resolved.extend(glob.glob(path))

    resolved = sorted(set(resolved))
    return resolved


def brute_force_divisions(
    paths: list[str],
) -> tuple[int, ...]:
    """Resolve the divisions of a potential array based on the parquet files directly

    Args:
        paths (list[str]): List of unique file paths (without wildcards) to loop over

    Returns:
        tuple[int]: The file sizes in the dask divisions schema, starting from zero

    Note:
        This method only works for Arrays whose partitions are all the file boundaries.
    """
    divisions: list[int] = [0]

    try:
        for file_path in paths:
            length_file: int = pq.ParquetFile(file_path).metadata.num_rows
            divisions.append(length_file)
    except ArrowInvalid as e:
        logger.error(f"Could not determine length of array:\n{e}")
    array = np.cumsum(np.array(divisions))
    return tuple(array.tolist())  # type: ignore


def brute_force_length(
    paths: list[str],
) -> int:
    """Last resort method to calculate the length of an array spanned over several files

    Args:
        paths (list[str]): List of unique file paths (without wildcards) to loop over

    Returns:
        int: The total length of the array
    """
    return brute_force_divisions(paths)[-1]


class NestedArrayIndexer:
    VALID_SEPARATORS = {".", "_", "/"}

    @classmethod
    def get_separator(cls, columns: list[str]) -> str:
        """
        Find the most common separator across all columns.

        Args:
            columns (list[str]): List of column names to analyze

        Returns:
            str: The most probable separator (".", "_", or "/")

        Raises:
            ValueError: If no valid separator is found
        """
        if not columns:
            raise ValueError("Cannot determine separator from empty column list")

        separator_counts = {sep: 0 for sep in cls.VALID_SEPARATORS}

        for column in columns:
            for separator in cls.VALID_SEPARATORS:
                if separator in column:
                    separator_counts[separator] += 1

        max_count = max(separator_counts.values())

        if max_count == 0:
            return ""

        most_common_seps = [sep for sep, count in separator_counts.items() if count == max_count]

        preference_order = [".", "_", "/"]  # tie breaker
        for preferred_sep in preference_order:
            if preferred_sep in most_common_seps:
                return preferred_sep

        return most_common_seps[0]

    @classmethod
    @overload
    def get_nested_field(cls, array: ak.Array, field: str, separator: str | None = None) -> ak.Array:
        ...

    @classmethod
    @overload
    def get_nested_field(cls, array: dak.Array, field: str, separator: str | None = None) -> dak.Array:
        ...

    @classmethod
    def get_nested_field(
        cls,
        array: dak.Array | ak.Array,
        field: str,
        separator: str | None = None,
    ) -> dak.Array | ak.Array:
        """Access a potentially nested field

        Args:
            array (dak.Array): The Dask Awkward Array to access.
            field (str): The field to access, which may contain nested fields separated by 'separator
            separator (str): The separator used to split the field into parts.

        Returns:
            dak.Array: The sub-array corresponding to the nested field.
        """
        if field in array.fields:
            return array[field]  # type: ignore
        try:
            return array[field]  # type: ignore
        except FieldNotFoundError:
            field_resolved = field if (separator is None) or (separator == "") else tuple(field.split(separator))
            return array[field_resolved]  # type: ignore

    @classmethod
    @overload
    def list_all_fields(cls, array: dak.Array, as_tuple: Literal[True]) -> list[tuple[str, ...]]:
        ...

    @classmethod
    @overload
    def list_all_fields(
        cls,
        array: dak.Array,
        as_tuple: Literal[False],
        separator: str,
    ) -> list[str]:
        ...

    @classmethod
    def list_all_fields(
        cls,
        array: dak.Array,
        as_tuple: bool = None,
        separator: str = ".",
    ) -> list[tuple[str, ...]] | list[str]:
        """
        Flattens the fields in a nested 'dask_awkward.Array' into a list of strings. Useful for
        files consisting of nested Records, such as the parquet files produced by ColumnFlow.

        Args:
            array (dak.Array): The Dask Awkward Array
            separator (str): Optional, the separator used to split nested fields, e.g. 'Lepton.pt'.
            as_tuple (bool): Optional, whether to return each field as a str with literal separator
                or as a tuple (without the separator). If True, the separator arg is only used internally.

        Returns:
            list[str]: A list of flattened field paths.
            list[tuple[str]]: A list of tuple of str that are compatible with ak.Array.__getitem__

        Raises:
            ValueError: If the input array does not have fields. In this case the
                error should be caught early. This is simply a safeguard.
        """
        if not hasattr(array, "fields"):
            raise ValueError("Input array does not have fields.")

        def recurse_fields(array: dak.Array, prefix: list[str] = None) -> list[str]:  # type: ignore
            if prefix is None:
                prefix = []

            if not hasattr(array, "fields"):
                return []

            paths: list[str] = []

            for field in array.fields:
                new_prefix = prefix + [field]
                sub_array = array[field]

                if hasattr(sub_array, "fields") and sub_array.fields:
                    paths.extend(recurse_fields(sub_array, new_prefix))
                else:
                    path_str = functools.reduce(lambda a, b: f"{a}{separator}{b}", new_prefix)
                    paths.append(path_str)

            return paths

        fields_as_str: list[str] = recurse_fields(array)

        if as_tuple:
            fields: list[tuple[str, ...]] = []

            for field in fields_as_str:
                fields.append(tuple(field.split(separator)))

            return fields  # return: list[tuple[str, ...]]
        else:
            return fields_as_str  # return: list[str]

    @classmethod
    def are_fields_nested(
        cls,
        array: dak.Array,
        columns: list[str] = None,
    ) -> bool:
        """Check if any field in a given array is nested

        A field is nested if it must be accessed by repeatedly calling array.<field_name>.<subfield_name>.
        This function ensures that all fields are either flat, e.g. can be indexed using 'array.fields' or
        nested, in which case they must later be accessed more carefully (for example with the NestedArrayIndexer
        class).

        Args:
            array (dak.Array): Input Dask Awkward Array
            columns (list[str] | None): List of columns to check. If None, all top-level fields in the array are
                checked.

        Returns:
            bool: True if all fields are nested, False if all fields are flat.

        Raises:
            ValueError: If the nestedness cannot be determined decidedly, for example if some fields are flat and
                others nested. Will indicate which fields are problematic.
        """
        if not columns:
            columns = array.fields

        is_field_nested: Literal[True, False, "MAYBE"]
        columns_nestedness = {column: None for column in columns}
        separator = cls.get_separator(columns)

        for column in columns:
            try:
                array[column]
                is_field_nested = False
                continue
            except (KeyError, FieldNotFoundError):
                is_field_nested = "MAYBE"

            try:
                cls.get_nested_field(array, column, separator)
                is_field_nested = True
            except (KeyError, FieldNotFoundError):
                is_field_nested = "MAYBE"
            finally:
                columns_nestedness[column] = is_field_nested  # type: ignore

        if "MAYBE" in columns_nestedness.values():
            problematic_columns = [col for col, is_nested in columns_nestedness.items() if is_nested == "MAYBE"]
            raise ValueError(
                "Cannot determine if fields are nested or not. Please specify the separator explicitly."
                f" Problematic columns are {problematic_columns}"
            )

        return all(columns_nestedness.values())
