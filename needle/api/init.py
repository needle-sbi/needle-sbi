from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Union

_TEMPLATES = Path(__file__).parent.parent / "templates"

_SETTINGS_JSON_TEMPLATE = """\
{
  "batch_system": "local",
  "htcondor_settings": {
    "request_memory": "2048MB",
    "request_cpus": 1,
    "+RequestRuntime": 3600
  },
  "slurm_settings": {
    "partition": "gpu",
    "time": "01:00:00",
    "mem": "4G",
    "cpus-per-task": 2
  }
}
"""


@dataclass
class InitResult:
    """Result of scaffolding a project via :func:`init`."""

    created: List[Path] = field(default_factory=list)
    skipped: List[Path] = field(default_factory=list)


def _copy(src: Path, dst: Path, description: str, result: InitResult, verbose: bool) -> None:
    label = src.name
    if dst.exists():
        if verbose:
            print(f"Skipped '{label}' ({description})")
        result.skipped.append(dst)
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        if verbose:
            print(f"Created '{label}' ({description})")
        result.created.append(dst)


def init(
    directory: Union[str, Path] = ".",
    *,
    no_conf: bool = False,
    backend: Literal["law", "b2luigi", "both"] = "both",
    verbose: bool = True,
) -> InitResult:
    """Scaffold a new NEEDLE project.

    Args:
        directory: Target directory (created if it does not exist).
        no_conf: Skip creating the `conf/` directory with default Hydra config groups.
        backend: Workflow backend to scaffold (`law.cfg`+`index` for ``"law"``,
            `settings.json` for ``"b2luigi"``, both for ``"both"``).
        verbose: Print a line for each created/skipped file (matches `needle init` CLI output).

    Returns:
        InitResult: paths created vs. skipped (already existed).
    """
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)

    result = InitResult()

    if backend in ("law", "both"):
        _copy(
            src=_TEMPLATES / "law.cfg",
            dst=target / "law.cfg",
            description="LAW config file for managing Tasks",
            result=result,
            verbose=verbose,
        )
        _copy(
            src=_TEMPLATES / "index",
            dst=target / "index",
            description="Index of needle.law_tasks, update with `law index`",
            result=result,
            verbose=verbose,
        )
    if backend in ("b2luigi", "both"):
        settings_dst = target / "settings.json"
        if settings_dst.exists():
            if verbose:
                print("Skipped 'settings.json' (b2luigi settings file)")
            result.skipped.append(settings_dst)
        else:
            settings_dst.write_text(_SETTINGS_JSON_TEMPLATE)
            if verbose:
                print("Created 'settings.json' (b2luigi settings file)")
            result.created.append(settings_dst)

    setup_dst = target / "setup.sh"
    _copy(
        src=_TEMPLATES / "setup.sh",
        dst=setup_dst,
        description="Setup script for setting up the NEEDLE environment",
        result=result,
        verbose=verbose,
    )
    if setup_dst.exists():
        setup_dst.chmod(0o755)

    if not no_conf:
        _copy(
            src=_TEMPLATES / "conf",
            dst=target / "conf",
            description="Config directory following the hydra schema",
            result=result,
            verbose=verbose,
        )

    return result
