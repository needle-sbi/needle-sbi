from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("api.run")


class UnknownTaskError(ValueError):
    """Raised when a requested b2luigi task name does not exist."""


@dataclass
class RunResult:
    """Result of submitting a task via :func:`run`.

    Attributes:
        returncode: The `law run` subprocess exit code for the ``law`` backend.
            Always ``None`` for the ``b2luigi`` backend, since
            ``b2luigi.cli.utils.process_task_instance()`` has no meaningful return value.
    """

    returncode: Optional[int]


ParamValue = Union[str, bool]


def _normalize_params(params: Union[Mapping[str, ParamValue], Sequence[str], None]) -> List[Tuple[str, ParamValue]]:
    """Normalize CLI-style ``"KEY=VALUE"``/bare-flag strings or a plain dict into
    a list of ``(key, value)`` pairs, where ``value is True`` means a bare flag.
    """
    if params is None:
        return []

    if isinstance(params, Mapping):
        return list(params.items())

    normalized: List[Tuple[str, ParamValue]] = []
    for param in params:
        key, sep, value = param.partition("=")
        normalized.append((key, value if sep else True))
    return normalized


def run(
    task: str = "MainTask",
    *,
    backend: Literal["law", "b2luigi"] = "b2luigi",
    config_file: str = "conf/config.yaml",
    results_path: str = "runs",
    batch_system: str = "local",
    workers: int = 1,
    params: Union[Mapping[str, ParamValue], Sequence[str], None] = None,
) -> RunResult:
    """Submit a needle task to a workflow backend from Python (e.g. a notebook or script).

    Args:
        task: Task class name to run, e.g. ``MainTask``, ``EnsembleTask``, ``TrainingTask``.
        backend: 
            - ``"law"`` shells out to ``law run`` and requires ``LAW_HOME``/``LAW_CONFIG_FILE``
            to already be set, e.g. by sourcing ``setup.sh`` or calling
                ``needle.api.configure_law()``.
            - ``"b2luigi"`` (the default) runs in-process via ``b2luigi.cli.utils.process_task_instance()``
                which is the same entry point the
                ``b2luigi run`` CLI command itself uses. This requires a ``tasks.py`` at the
                project root. The ``b2luigi`` console script must be on ``PATH`` on worker nodes too.
        config_file: Path to the Hydra config file. Defaults to `conf/config.yaml`
        results_path: Root directory for results. Defaults to `runs`
        batch_system: One of ``"local"``, ``"htcondor"``, ``"slurm"``, ``"lsf"`` (b2luigi only).
        workers: Number of parallel workers (b2luigi only).
        params: Extra task parameters. Either a ``dict`` (native Python callers) or a list of
            ``"KEY=VALUE"`` and bare-flag strings (CLI-style).

    Returns:
        RunResult: the subprocess return code (law) or ``None`` (b2luigi).

    Raises:
        UnknownTaskError: if the Task name is not a known b2luigi task class name.
    """
    normalized_params = _normalize_params(params)

    if backend == "law":
        logger.info("Running with `law` workflow backend")

        law_args = ["law", "run", task]
        if config_file:
            law_args += ["--config-file", config_file]
        if results_path:
            law_args += ["--results-path", results_path]
        for key, value in normalized_params:
            flag = f"--{key.replace('_', '-')}"
            law_args += [flag, value] if value is not True else [flag]

        return RunResult(returncode=subprocess.call(law_args))

    elif backend == "b2luigi":
        from b2luigi.cli.utils import process_task_instance

        import needle.tasks.b2luigi as b2luigi_tasks
        from needle.tasks.b2luigi.workflows.common import configure_b2luigi

        logger.info("Running with `b2luigi` workflow backend")

        task_cls = getattr(b2luigi_tasks, task, None)
        if task_cls is None:
            available = ", ".join(b2luigi_tasks.__all__)
            raise UnknownTaskError(f"Unknown b2luigi task '{task}'. Available: {available}")

        resolved_config_file = config_file if config_file is not None else "conf/config.yaml"
        resolved_results_path = results_path if results_path is not None else "runs"

        extra_params: Dict[str, ParamValue] = dict(normalized_params)

        configure_b2luigi(batch_system=batch_system)

        task_instance = task_cls(
            config_file=resolved_config_file,
            results_path=resolved_results_path,
            **extra_params,
        )

        process_task_instance(
            task_instance,
            task_file="tasks.py",
            workers=workers,
            batch=batch_system != "local",
        )

        return RunResult(returncode=None)

    raise ValueError(f"Unknown backend '{backend}'. Expected 'law' or 'b2luigi'.")
