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
    """Add the settings for a Workflow from the law.cfg to the job Config

    Note:
        Law will pass through luigi configs when they are labelled `luigi_<section>`. Therefore, our
        Workflow is accessible through the section `[luigi_<Task>_<batch_system>]`.

    Args:
        self (SupportsLuigiAPI): Any Task that inherits from `luigi.Task`
        cfg (RemoteConfig): The config used by the Workflow. One of `htcondor.HTCondorJobFileFactory.Config`
            or `slurm.SlurmJobFileFactory.Config` depending on the Workflow.
        workflow_type (Literal["htcondor", "slurm"]): The name of the batch system
            to use. This is used for accessing the correct section in the luigi cfg.

    Raises:
        ValueError: If the sub-config for luigi does not contain the proper section. If the section
            exists but is empty, then only a Warning is triggered

    Returns:
        RemoteConfig: The same object as `cfg` but with the added items from the luigi cfg.
    """
    luigi_cfg: LuigiConfig = luigi.configuration.get_config()
    section = f"{self.get_task_family()}_{workflow_type}"

    if luigi_cfg.has_section(section):
        if not luigi_cfg.items(section):
            logger.warning(f"The law.cfg section '[luigi_{section}]' is empty.")

        for key, value in luigi_cfg.items(section):
            cfg.custom_content.append((key, value))
    else:
        raise ValueError(
            f"Your 'law.cfg' file does not contain a '[luigi_{section}]' section. "
            f"Add it in the following format:\n"
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
