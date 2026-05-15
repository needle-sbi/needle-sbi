"""
Convert root to parquet files using uproot and Awkward
"""
from collections import defaultdict
from pathlib import Path
from typing import List

import awkward as ak
import uproot
from tqdm import tqdm

from needle.etl.array import NestedArrayIndexer, resolve_paths


def convert_root_to_parquet(
    input_paths: str | Path,
    output_dir: str | Path,
    drop_branches: List[str] = ["fBits"],
    row_group_size: int | None = 1_000_000,
    step_size: str = "1 GB",
    output_file_basename: str = "events",
    test_mode: bool = False,
) -> None:
    """Convert a set of root files to equivalently represented Awkward Arrays in a parquet file.
    Useful for benchmarking purposes as the structure of the arrays remains the same between both
    file formats.

    Important:
        This conversion function assumes that there is only one level of nestedness (most common
        case in HEP). For example, columns like `Photon.Eta` will be correctly reproduced in the
        parquet file, while `GenJet.SoftDroppedJet.ref` will fail. In order to skip these columns,
        use the `drop_branches` argument to blacklist invalid names.

    Args:
        input_paths (str | Path): A str or glob pattern to a set of root files. This is passed to
            `uproot.iterate`. The canonical format would be `/path/to/files_*.root/` for example.
        output_dir (str | Path): A str or Path pointing to the directory where the output parquet
            files should be created. Assumes that this directory already exists. The output name is
            then decided by the `output_file_base_name` argument.
        drop_branches (List[str], optional): Which keywords to blacklist when going through the
            branches. If any of the str in the provided list matches a substring in the name of the
            branch, that branch will be dropped. For example, adding `"Genjet"` will remove all
            branches containing that keyword. Defaults to ["fBits"], since that column often breaks
            uproot when it tries to read Delphes-schema files.
        row_group_size (int | None, optional): Size of the row_groups in the final parquet file.
            Defaults to None, which reduces to the `ak.to_parquet` default.
        step_size (str, optional): Cache size for `uproot.iterate`. Defaults to "1 GB".
        output_file_basename (str, optional): Name base for the parquet files. Defaults to "events",
            with all files being named `events_0`... `events_<N>` where N is the number of root
            files.
        test_mode (bool, optional): Whether to only process a single root file. Defaults to False.
    """

    def filter_name_func(name: str) -> bool:
        return not any(d in name for d in drop_branches)

    total_num_files = len(resolve_paths(str(input_paths)))

    if all([(Path(output_dir) / f"{output_file_basename}_{i}.parquet").exists() for i in range(total_num_files)]):
        return None

    for i, array in tqdm(
        enumerate(
            uproot.iterate(
                files=input_paths,
                filter_name=filter_name_func,
                step_size=step_size,
            )
        ),
        total=total_num_files,
    ):
        output_path = Path(output_dir) / f"{output_file_basename}_{i}.parquet"

        if output_path.exists():
            continue

        groups = defaultdict(dict)
        columns = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")  # type: ignore

        for column in columns:
            if not filter_name_func(column) or "." not in column:
                continue

            prefix, subfield = column.split(".", 1)
            groups[prefix][subfield] = array[column]  # type: ignore

        out_dict = {}

        for column, subarray in groups.items():
            if isinstance(subarray, dict):
                out_dict[column] = ak.zip(subarray, depth_limit=1)
            else:
                out_dict[column] = subarray

        out_array = ak.Array(out_dict)
        ak.to_parquet(out_array, output_path, row_group_size=row_group_size)

        if test_mode:
            break
