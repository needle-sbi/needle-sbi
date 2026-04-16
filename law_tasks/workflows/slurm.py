import os
from typing import List

import law
from law.contrib import slurm
from law.util import rel_path
from preprocessor.utils.logging import ColorFormatter
from law_tasks.workflows.common import get_script_dir, add_workflow_settings_from_cfg, Config


logger = ColorFormatter.get_logger("slurm")


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
        config = super().slurm_job_config(config, job_num, branches)
        config = add_workflow_settings_from_cfg(self, config, workflow_type="slurm")

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(get_script_dir(), "pyproject.toml"),
        )
        config.input_files["setup.sh"] = law.JobInputFile(
            os.path.join(get_script_dir(), "setup.sh"),
        )

        config.stdout = "stdout_%j.txt"  # %j = Slurm job id
        config.stderr = "stderr_%j.txt"

        return config
