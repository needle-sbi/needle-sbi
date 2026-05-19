import os
from typing import List

import law
from law.contrib import slurm

from needle.law_tasks.workflows.common import (
    RemoteConfig,
    add_workflow_settings_from_cfg,
    get_script_dir,
)
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("slurm")


class SlurmWorkflow(slurm.SlurmWorkflow):
    results_path: str

    def slurm_output_directory(self) -> law.LocalDirectoryTarget:  # type: ignore
        return law.LocalDirectoryTarget(
            os.path.join(get_script_dir(), self.results_path, "slurm", self.__class__.__name__),
        )

    def slurm_job_config(
        self,
        config: RemoteConfig,
        job_num: int,
        branches: List[int],
    ):
        config = super().slurm_job_config(config, job_num, branches)
        config = add_workflow_settings_from_cfg(self, config, workflow_type="slurm")

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(get_script_dir(), "pyproject.toml"),
        )
        config.input_files["setup.sh"] = law.JobInputFile(
            os.path.join(get_script_dir(), "setup.sh"),
        )

        config.custom_content.append(("export", "ALL"))
        config.render_variables["script_dir"] = get_script_dir()

        config.stdout = "stdout_%j.txt"  # %j = Slurm job id
        config.stderr = "stderr_%j.txt"
        config.log = "slurm.log"

        return config
