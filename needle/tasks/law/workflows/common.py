from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal, Protocol, Union

import luigi
from law.contrib import htcondor, slurm

from needle.utils.logging import ColorFormatter

#: Configuration dataclass for HTCondor job submission parameters.
HTCondorConfig = htcondor.HTCondorJobFileFactory.Config
#: Configuration dataclass for SLURM job submission parameters.
SlurmConfig = slurm.SlurmJobFileFactory.Config
#: Union of all supported remote execution backend configurations.
RemoteConfig = Union[HTCondorConfig, SlurmConfig]
#: Luigi configuration parser used to read ``luigi.cfg`` settings at runtime.
LuigiConfig = luigi.configuration.cfg_parser.LuigiConfigParser


class SupportsLuigiAPI(Protocol):
    def get_task_family(self) -> str:
        """Implements `luigi.Task.get_task_family` which returns the name of the Task."""
        ...

    #: Per-instance batch resource dict provided by `BaseTrainingTask`/`BaseDownstreamMixin`.
    #: Deliberately not named `resources` - that name is reserved by `luigi.Task` for its own
    #: scheduler resource-pool accounting; reusing it here would hang the worker.
    batch_resources: dict


logger = ColorFormatter.get_logger("workflow")


def get_script_dir() -> str:
    """Find the root directory of the project.

    Uses the `$SCRIPT_DIR` environment variable when set (exported by `setup.sh` for cloned-repo
    usage). Falls back to the current working directory, which is correct when the package is
    installed via pip and law is invoked from the user's project directory.

    Returns:
        str: The path to the root directory of the project.
    """
    _script_dir = os.getenv("SCRIPT_DIR")
    return _script_dir if _script_dir else str(Path.cwd())


def add_workflow_settings_from_cfg(
    self: SupportsLuigiAPI,
    cfg: RemoteConfig,
    workflow_type: Literal["htcondor", "slurm"],
) -> RemoteConfig:
    """Add batch resource settings to the job Config.

    Prefers ``self.batch_resources`` (a plain dict, e.g. from ``EstimatorConfig.resources`` /
    ``DownstreamTaskConfig.resources``, merged with the active systematic's resources where
    applicable) when it is non-empty - keys/values are forwarded verbatim, unvalidated.

    Note:
        When ``self.batch_resources`` is empty/unset, falls back to the legacy mechanism: law
        passes through luigi configs labelled `luigi_<section>`, so a Task's settings can also be
        read from the section `[luigi_<Task>_<batch_system>]` in `law.cfg`.

    Args:
        self (SupportsLuigiAPI): Any Task that inherits from `luigi.Task`
        cfg (RemoteConfig): The config used by the Workflow. One of `htcondor.HTCondorJobFileFactory.Config`
            or `slurm.SlurmJobFileFactory.Config` depending on the Workflow.
        workflow_type (Literal["htcondor", "slurm"]): The name of the batch system
            to use. This is used for accessing the correct section in the luigi cfg.

    Raises:
        ValueError: If ``self.batch_resources`` is empty/unset and the law.cfg fallback section
            is missing. If the section exists but is empty, only a Warning is triggered.

    Returns:
        RemoteConfig: The same object as `cfg` but with the added resource settings.
    """
    if self.batch_resources:
        for key, value in self.batch_resources.items():
            cfg.custom_content.append((key, value))
        return cfg

    luigi_cfg: LuigiConfig = luigi.configuration.get_config()
    section = f"{self.get_task_family()}_{workflow_type}"

    if luigi_cfg.has_section(section):
        if not luigi_cfg.items(section):
            logger.warning(f"The law.cfg section '[luigi_{section}]' is empty.")

        for key, value in luigi_cfg.items(section):
            cfg.custom_content.append((key, value))
    else:
        raise ValueError(
            f"No 'resources' were set on '{self.get_task_family()}' and your 'law.cfg' file does "
            f"not contain a '[luigi_{section}]' section either. Add resources to your config's "
            f"estimator/systematic/downstream_task entry, or add a section to 'law.cfg' in the "
            f"following format:\n"
            f"    [luigi_{section}]\n"
            f"    nodes: 1  # for example\n"
            f"    ...\n"
            f"Available luigi sections are: {luigi_cfg.sections()}\n"
        )

    return cfg


def check_batch_system(system: Literal["local", "htcondor", "slurm"]) -> None:
    """Ensure that the flag set by the user for `workflow=<system>` actually matches a valid batch
    system. Otherwise the error produced by law is rather cryptic and difficult to understand.

    Args:
        system (Literal["local", "htcondor", "slurm]): The batch system name to check.
            Currently only local, htcondor and slurm are supported.

    Raises:
        RuntimeError: If the batch system is not available from using `shutil.which`
        ValueError: If the batch system is not in the list of available systems
    """

    valid_batch_systems = {
        "local": "law",
        "htcondor": "condor_submit",
        "slurm": "sbatch",
    }

    binary = valid_batch_systems.get(system)

    if binary:
        if shutil.which(binary) is None:
            logger.error(f"Selected batch system '{system}' is not available: '{binary}' not in PATH. ")
    else:
        logger.warning(f"Selected batch system '{system}' is not in {list(valid_batch_systems.keys())}")
