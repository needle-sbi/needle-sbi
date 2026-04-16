import os
from pathlib import Path
from law.contrib import htcondor
from law.contrib import slurm
import luigi
from typing import Union, Literal, Protocol
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
    _script_dir = os.getenv("SCRIPT_DIR")

    if not _script_dir:
        _script_dir = Path(os.path.abspath(__file__)).parent.parent.parent

    _script_dir = Path(_script_dir) if isinstance(_script_dir, str) else _script_dir

    if not _script_dir.name == "orchestrator":
        raise ValueError(
            "The path to the root directory of the project should end with 'orchestrator' "
            f"but is {_script_dir}"
        )

    return str(_script_dir)


def add_workflow_settings_from_cfg(
    self: SupportsLuigiAPI,
    cfg: Config,
    workflow_type: Literal["htcondor", "slurm"],
) -> Config:
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


def check_batch_system(system: Literal["htcondor", "slurm"]) -> None:
    import shutil

    valid_batch_systems = {
        "htcondor": "condor_submit",
        "slurm": "sbatch",
    }

    binary = valid_batch_systems.get(system)

    if binary:
        if shutil.which(binary) is None:
            raise RuntimeError(
                f"Selected batch system '{system}' is not available: '{binary}' not in PATH. "
            )
    else:
        raise RuntimeError(f"Selected batch system '{system}' is not in {list(valid_batch_systems.keys())}")