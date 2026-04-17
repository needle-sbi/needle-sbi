import os
import shutil
from pathlib import Path
from typing import Literal, Protocol, Union

import luigi
from law.contrib import htcondor, slurm

from preprocessor.utils.logging import ColorFormatter

HTCondorConfig = htcondor.HTCondorJobFileFactory.Config
SlurmConfig = slurm.SlurmJobFileFactory.Config
Config = Union[HTCondorConfig, SlurmConfig]
LuigiConfig = luigi.configuration.cfg_parser.LuigiConfigParser


class SupportsLuigiAPI(Protocol):
    def get_task_family(self) -> str:
        ...


logger = ColorFormatter.get_logger("workflow")


def get_script_dir() -> str:
    """Find the path to the root directory of the project. Either by accessing the environment variable
    `$SCRIPT_DIR` which is defined in the `setup.sh` file or by calculating the relative path based
    on the location of this file.

    Raises:
        ValueError: If the resulting path does not end with 'orchestrator', which is the hard-coded
            name of the project.

    Returns:
        str: The path to the root directory of the project.
    """
    _script_dir = os.getenv("SCRIPT_DIR")

    if not _script_dir:
        _script_dir = Path(os.path.abspath(__file__)).parent.parent.parent

    _script_dir = Path(_script_dir) if isinstance(_script_dir, str) else _script_dir

    if not _script_dir.name == "orchestrator":
        raise ValueError(
            "The path to the root directory of the project should end with 'orchestrator' " f"but is {_script_dir}"
        )

    return str(_script_dir)


def add_workflow_settings_from_cfg(
    self: SupportsLuigiAPI,
    cfg: Config,
    workflow_type: Literal["htcondor", "slurm"],
) -> Config:
    """Add the settings for a Workflow from the law.cfg to the job Config

    Note:
        Law will pass through luigi configs when they are labelled "luigi_<section>". Therefore, our
        Workflow is accessible through the section "[luigi_<Task>_<batch_system>]".

    Args:
        self (SupportsLuigiAPI): Any Task that inherits from luigi
        cfg (Config): The Config used by the Workflow. One of `htcondor.HTCondorJobFileFactory.Config`
            or `slurm.SlurmJobFileFactory.Config` depending on the Workflow.
        workflow_type (Literal[&quot;htcondor&quot;, &quot;slurm&quot;]): The name of the batch system
            to use. This is used for accessing the correct section in the luigi cfg.

    Raises:
        ValueError: If the sub-config for luigi does not contain the proper section. If the section
            exists but is empty, then only a Warning is triggered

    Returns:
        Config: The same object as `cfg` but with the added items from the luigi cfg.
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
            raise RuntimeError(f"Selected batch system '{system}' is not available: '{binary}' not in PATH. ")
    else:
        raise ValueError(f"Selected batch system '{system}' is not in {list(valid_batch_systems.keys())}")
