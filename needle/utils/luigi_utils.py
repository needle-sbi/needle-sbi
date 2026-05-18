from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, overload

import law
import luigi

#: A single Luigi ``LocalTarget`` file handle.
LuigiTarget = luigi.LocalTarget
#: A single LAW ``LocalFileTarget`` file handle.
LawTarget = law.LocalFileTarget

#: Any form a Luigi target collection can take: a single target, list, dict, or tuple.
LuigiTargetCollection = Union[
    LuigiTarget,
    List[LuigiTarget],
    Dict[str, LuigiTarget],
    Tuple[LuigiTarget, ...],
]

#: Any form a LAW target collection can take: a single target, list, dict, or tuple.
LawTargetCollection = Union[
    LawTarget,
    List[LawTarget],
    Dict[str, LawTarget],
    Tuple[LawTarget, ...],
]


def _convert_single_luigi_to_law_target(target: luigi.LocalTarget) -> law.LocalFileTarget:
    """Convert a Luigi LocalTarget to a law LocalFileTarget. If already a law LocalFileTarget, the
    input target is simply returned. Otherwise create a new law LocalFileTarget from scratch.

    Args:
        target: A Luigi LocalTarget instance to be converted to a law LocalFileTarget.

    Returns:
        law.LocalFileTarget

    Warns:
        warnings: If the target is a temporary Luigi target, a warning is issued
            indicating that conversion may be unpredictable.
    """
    if isinstance(target, law.LocalFileTarget):
        return target
    if target.is_tmp:
        import warnings

        warnings.warn(
            f"Converting a temporary luigi.LocalTarget at {target.path!r} may be unpredictable.",
            stacklevel=3,
        )
    return law.LocalFileTarget(
        path=Path(target.path).absolute(),
        is_tmp=target.is_tmp,
    )


@overload
def convert_luigi_to_law_targets(
    luigi_targets: LuigiTarget,
) -> LawTarget:
    ...


@overload
def convert_luigi_to_law_targets(
    luigi_targets: List[LuigiTarget],
) -> List[LawTarget]:
    ...


@overload
def convert_luigi_to_law_targets(
    luigi_targets: Dict[str, LuigiTarget],
) -> Dict[str, LawTarget]:
    ...


@overload
def convert_luigi_to_law_targets(
    luigi_targets: Tuple[LuigiTarget, ...],
) -> Tuple[LawTarget, ...]:
    ...


def convert_luigi_to_law_targets(
    luigi_targets: LuigiTargetCollection,
) -> LawTargetCollection:
    """Convert Luigi targets to Law targets.

    Takes a Luigi target or collection of targets and converts them to their
    corresponding Law target equivalents.

    Args:
        luigi_targets:
            - single `luigi.LocalTarget `instance
            - `list` of` luigi.LocalTarget` instances
            - `dict` mapping keys to `luigi.LocalTarget` instances with None values filtered out
            - `tuple` of `luigi.LocalTarget`

    Returns:
        LawTargetCollection: Same collection as the input, just with law instead of luigi targets.

    Raises:
        TypeError: If luigi_targets is not a LocalTarget, list, or dict.
    """
    if isinstance(luigi_targets, luigi.LocalTarget):
        return _convert_single_luigi_to_law_target(luigi_targets)
    if isinstance(luigi_targets, list):
        return [_convert_single_luigi_to_law_target(target) for target in luigi_targets]
    if isinstance(luigi_targets, dict):
        return {
            key: _convert_single_luigi_to_law_target(target)
            for key, target in luigi_targets.items()
            if target is not None
        }
    if isinstance(luigi_targets, tuple):
        return tuple(_convert_single_luigi_to_law_target(target) for target in luigi_targets)
    raise TypeError(f"Target(s) of type: {type(luigi_targets)} must be `LocalTarget`, list or dict.")


def collect_output_paths(
    task: luigi.Task | law.Task,
    visited: Set[str] = None,
    current_depth: int = 0,
    max_depth: int = -1,
) -> List[str]:
    visited = visited or set()

    task_id = task.task_id

    if task_id in visited:
        return []

    visited.add(task_id)

    paths = []

    for target in luigi.task.flatten(task.output()):
        if hasattr(target, "path"):
            paths.append(getattr(target, "path"))

    if max_depth < 0 or current_depth < max_depth:
        for dep in luigi.task.flatten(task.requires()):
            paths.extend(
                collect_output_paths(
                    task=dep,  # type: ignore
                    visited=visited,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                )
            )

    return paths
