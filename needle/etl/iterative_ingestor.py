"""
Non-dask Ingestors. Supports parquet and root files. No virtual arrays are used, the class iterates
over chunks of data yielding all fields at once. This is a more typical approach to event loading.

Use when the full lazy graph construction cost is not worth paying. For example, a single streaming
of many large root files.

Instantiation is same as the dask-based Ingestor. The iteration happens by chunk and not by column.
See `needle.etl.protocols` for the Protocol shared by the Ingestors.
"""

import reprlib
from typing import Any, Callable, Iterator, Literal

import awkward as ak
import pyarrow.parquet as pq
import pydantic
import uproot

from needle.etl.array import (
    NestedArrayIndexer,
    brute_force_divisions,
    check_columns_found,
    resolve_input_format,
    resolve_paths,
)
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("etl", level="DEBUG")


class IterableIngestor:
    """Stream input files chunk-by-chunk.

    Note:
        Currently supported formats:
            - `parquet`: Read one file at a time using `awkward.from_parquet`.
            - `root`: Stream fixed-size steps across all files using `uproot.iterate`.

    Attributes:
        paths (list[str]): Resolved input file paths (glob patterns already expanded).
        format (Literal["parquet", "root"]): Resolved input file format.
        treename (str | None): Name of the `TTree` read from each file. Always `None` for
            `format == "parquet"`.
        fields (list[str]): List of fields that will be read.
        num_classes (int): Number of fields in `fields`.
        length (int): Number of events across all input files.
        divisions (tuple[int, ...]): Cumulative per-file event-count boundaries, starting at 0 --
            the same schema as `dask_awkward.Array.divisions` / `needle.etl.array.brute_force_divisions`.
        step_size (str | int): Chunk size forwarded to `uproot.iterate`, either a number of
            entries or a memory-size string with units (e.g. `"100 MB"`). Unused for
            `format == "parquet"`.
        max_number_events (int): Cap on the total number of events `iterate()` will yield, or `-1`
            for no cap.
        SEPARATOR (str): Separator used for nested fields. Default is '.'.

    Important:
        New methods for reading other formats must implement the following:
        - Be added to the `__init__` method
        - Listed as a supported format in the `format` argument
        - Added to the `format == "automatic"` clause in `VALID_FORMATS`
        - Given a corresponding branch in `iterate()`
    """

    paths: list[str]
    format: Literal["parquet", "root"]
    treename: str | None
    fields: list[str]
    num_classes: int
    length: int
    divisions: tuple[int, ...]
    step_size: str | int
    max_number_events: int
    SEPARATOR: str = "."
    VALID_FORMATS = {"parquet", "root"}

    @pydantic.validate_call
    def __init__(
        self,
        paths: str | list[str],
        format: Literal["parquet", "root", "automatic"] = "automatic",
        columns: str | list[str] | None = None,
        treename: str | None = None,
        step_size: str | int = "1 GB",
        max_number_events: int = -1,
    ) -> None:
        """
        Resolve input files, the format, the fields to read, and the total length. No event
        data is read at this stage.

        Args:
            paths (str | list[str]): Path(s) or glob pattern(s) to the input file(s).
            format (Literal["parquet", "root", "automatic"]): Format of the input file(s).
                If `'automatic'`, the format will be inferred based on the file extension.
            columns (str | list[str] | None): List of columns to read from the input file(s).
                If None, all columns found in a representative file are used.
            treename (str | None): Name of the `TTree` to read. Only used when `format` resolves
                to `"root"`. If None, it is auto-detected from the first file (requires exactly
                one `TTree` per file).
            step_size (str | int): Chunk size passed to `uproot.iterate` when `format` resolves
                to `"root"`. Ignored for `"parquet"`.
            max_number_events (int): If positive, `iterate()` stops (truncating the last chunk if
                necessary) once this many events have been yielded in total. `-1` (default) means
                no limit.

        Returns:
            IterableIngestor: self, with `paths`/`format`/`fields`/`length` resolved.

        Raises:
            FileNotFoundError: If no files match `paths`.
        """
        paths = [paths] if isinstance(paths, str) else paths
        columns = [columns] if isinstance(columns, str) else columns

        self.paths = resolve_paths(paths)

        if not self.paths:
            raise FileNotFoundError(f"No files could be found with pattern {paths}")

        self.format = resolve_input_format(format, self.paths[0], self.VALID_FORMATS)  # type: ignore[assignment]
        self.step_size = step_size
        self.max_number_events = max_number_events

        match self.format:
            case "root":
                self.treename = treename or self._detect_treename(self.paths[0])
                self.divisions = brute_force_divisions(self.paths, file_type="root", treename=self.treename)
                loaded_columns = self._resolve_fields_root(self.paths[0], self.treename)
            case "parquet":
                self.treename = None
                self.divisions = brute_force_divisions(self.paths, file_type="parquet")
                loaded_columns = self._resolve_fields_parquet(self.paths[0])

        check_columns_found(columns or [], loaded_columns)
        self.fields = columns or loaded_columns
        self.num_classes = len(self.fields)
        self.length = self.divisions[-1]

        if max_number_events > 0:
            self.length = min(self.length, max_number_events)

        logger.info(f"Resolved {self.length} events with {self.num_classes} column(s): {reprlib.repr(self.fields)}")
        return None

    @staticmethod
    def _detect_treename(path: str) -> str:
        """Find the single `TTree` in a file without reading any branch data.

        Args:
            path (str): Representative file path to inspect.

        Returns:
            str: Name of the single `TTree` found in `path`.

        Raises:
            ValueError: If `path` does not contain exactly one `TTree`.
        """
        with uproot.open(path) as file:  # type: ignore
            tree_names = file.keys(filter_classname="TTree", cycle=False)

        if len(tree_names) != 1:
            raise ValueError(
                f"Could not auto-detect a single TTree in '{path}' (found {tree_names}). Pass 'treename' explicitly."
            )
        return tree_names[0]

    @staticmethod
    def _resolve_fields_root(path: str, treename: str) -> list[str]:
        with uproot.open(path) as file:  # type: ignore
            return list(file[treename].keys(full_paths=False))  # type: ignore

    def _resolve_fields_parquet(self, path: str) -> list[str]:
        empty_array = ak.from_arrow(pq.ParquetFile(path).schema_arrow.empty_table())
        return NestedArrayIndexer.list_all_fields(empty_array, as_tuple=False, separator=self.SEPARATOR)

    def _filter_name(self) -> Callable[[str], bool]:
        """Build the `filter_name` function passed to `uproot.iterate`"""
        fields = self.fields

        def _filter(name: str) -> bool:
            return name in fields

        return _filter

    def iterate(self, **reader_kwargs: Any) -> Iterator[ak.Array]:
        """Stream chunks of data, one `root` step or `parquet` file at a time.

        Args:
            **reader_kwargs: Forwarded to the underlying reader:
                - `uproot.iterate` (for `"root"` overriding `step_size` if given)
                - `awkward.from_parquet` (for `"parquet"`)

        Yields:
            ak.Array: Successive chunks of events covering every field in `fields`, in file
                order. The final chunk is truncated if `max_number_events` was set and would
                otherwise be exceeded.
        """
        if self.format == "root":
            chunks = self._iterate_root(**reader_kwargs)
        else:
            chunks = self._iterate_parquet(**reader_kwargs)

        events_yielded = 0

        for chunk in chunks:
            if self.max_number_events > 0:
                remaining = self.max_number_events - events_yielded
                if remaining <= 0:
                    return
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]

            events_yielded += len(chunk)
            yield chunk

    def _iterate_root(self, **reader_kwargs: Any) -> Iterator[ak.Array]:
        options = {"step_size": self.step_size, **reader_kwargs}
        files = [f"{path}:{self.treename}" for path in self.paths]
        yield from uproot.iterate(files=files, filter_name=self._filter_name(), **options)  # type: ignore

    def _iterate_parquet(self, **reader_kwargs: Any) -> Iterator[ak.Array]:
        for path in self.paths:
            yield ak.from_parquet(path, columns=self.fields, **reader_kwargs)
