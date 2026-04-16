import os
from pathlib import Path
from typing import List

import luigi
import law
from law.contrib import slurm
from law.util import rel_path
from preprocessor.utils.logging import ColorFormatter


logger = ColorFormatter.get_logger("slurm")

Config = slurm.SlurmJobFileFactory.Config
LuigiConfig = luigi.configuration.cfg_parser.LuigiConfigParser


class SlurmWorkflow(slurm.SlurmWorkflow):
    def slurm_output_directory(self) -> law.LocalDirectoryTarget:  # type: ignore
        return law.LocalDirectoryTarget(os.path.join("/tmp/law_output", "slurm", self.__class__.__name__))

    def slurm_bootstrap_file(self):  # type: ignore
        bootstrap_file = os.path.join(rel_path(__file__), "bootstrap.sh")
        return law.JobInputFile(bootstrap_file, share=True, render_job=True)

    def slurm_job_config(
        self,
        config: Config,
        job_num: int,
        branches: List[int],
    ):
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

        def add_slurm_settings(cfg: Config) -> Config:
            luigi_cfg: LuigiConfig = luigi.configuration.get_config()
            section = f"{self.get_task_family()}_slurm"

            if luigi_cfg.has_section(section):
                if not luigi_cfg.items(section):
                    logger.warning(f"The law.cfg section [luigi_{self.get_task_family()}_slurm] is empty.")

                for key, value in luigi_cfg.items(section):
                    cfg.custom_content.append((key, value))
            else:
                raise ValueError(
                    f"Your 'law.cfg' file does not contain a 'luigi_{self.get_task_family()}_slurm' section. "
                    f"Add it in the following format:\n"
                    f"    [{self.get_task_family()}_slurm]\n"
                    f"    nodes: 1  # for example\n"
                    f"    ...\n"
                    f"Available luigi sections are: {luigi_cfg.sections()}\n"
                )

            return cfg

        config = super().slurm_job_config(config, job_num, branches)

        config = add_slurm_settings(config)

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(get_script_dir(), "pyproject.toml"),
        )
        config.input_files["setup.sh"] = law.JobInputFile(
            os.path.join(get_script_dir(), "setup.sh"),
        )

        config.stdout = "stdout_%j.txt"  # %j = Slurm job id
        config.stderr = "stderr_%j.txt"

        return config
