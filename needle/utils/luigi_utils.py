from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union, overload

import luigi

if TYPE_CHECKING:
    import law

#: Any form a Luigi target collection can take: a single target, list, dict, or tuple.
LuigiTarget = luigi.LocalTarget
LuigiTargetCollection = Union[
    LuigiTarget,
    List[LuigiTarget],
    Dict[str, LuigiTarget],
    Tuple[LuigiTarget, ...],
]

#: Any form a Law target collection can take: a single target, list, dict, or tuple.
LawTargetCollection = Union[
    "law.LocalFileTarget",
    "List[law.LocalFileTarget]",
    "Dict[str, law.LocalFileTarget]",
    "Tuple[law.LocalFileTarget, ...]",
]


def _to_law_file_target(target: luigi.LocalTarget):  # type: ignore[return]
    """Convert a single ``luigi.LocalTarget`` to ``law.LocalFileTarget``.

    Importing law is deferred so that this module remains importable without LAW
    installed (e.g. when using the b2luigi backend exclusively).
    """
    import law

    if isinstance(target, law.LocalFileTarget):
        return target
    if target.is_tmp:
        warnings.warn(
            f"Converting a temporary luigi.LocalTarget at {target.path!r} may be unpredictable.",
            stacklevel=3,
        )
    return law.LocalFileTarget(
        path=Path(target.path).absolute(),
        is_tmp=target.is_tmp,
    )


@overload
def convert_luigi_to_law_targets(luigi_targets: LuigiTarget) -> "law.LocalFileTarget": ...  # type: ignore[name-defined]


@overload
def convert_luigi_to_law_targets(luigi_targets: List[LuigiTarget]) -> "List[law.LocalFileTarget]": ...  # type: ignore[name-defined]


@overload
def convert_luigi_to_law_targets(luigi_targets: Dict[str, LuigiTarget]) -> "Dict[str, law.LocalFileTarget]": ...  # type: ignore[name-defined]


@overload
def convert_luigi_to_law_targets(luigi_targets: Tuple[LuigiTarget, ...]) -> "Tuple[law.LocalFileTarget, ...]": ...  # type: ignore[name-defined]


def convert_luigi_to_law_targets(luigi_targets: LuigiTargetCollection) -> LawTargetCollection:
    """Convert Luigi targets to Law targets.

    Takes a Luigi target or collection of targets and converts them to their
    corresponding Law target equivalents.

    Args:
        luigi_targets:
            - single ``luigi.LocalTarget`` instance
            - ``list`` of ``luigi.LocalTarget`` instances
            - ``dict`` mapping keys to ``luigi.LocalTarget`` instances with None values filtered out
            - ``tuple`` of ``luigi.LocalTarget``

    Returns:
        LawTargetCollection: Same collection structure, with ``law.LocalFileTarget`` instances.

    Raises:
        TypeError: If luigi_targets is not a LocalTarget, list, dict, or tuple.
    """
    if isinstance(luigi_targets, luigi.LocalTarget):
        return _to_law_file_target(luigi_targets)
    if isinstance(luigi_targets, list):
        return [_to_law_file_target(target) for target in luigi_targets]
    if isinstance(luigi_targets, dict):
        return {
            key: _to_law_file_target(target)
            for key, target in luigi_targets.items()
            if target is not None
        }
    if isinstance(luigi_targets, tuple):
        return tuple(_to_law_file_target(target) for target in luigi_targets)
    raise TypeError(f"Target(s) of type: {type(luigi_targets)} must be `LocalTarget`, list, dict, or tuple.")


def collect_output_paths(
    task: luigi.Task,
    visited: Set[str] | None = None,
    current_depth: int = 0,
    max_depth: int = -1,
) -> List[str]:
    """Recursively collect all output file paths from a task and its dependencies.

    Works with any ``luigi.Task`` subclass (including LAW and b2luigi tasks).
    """
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
