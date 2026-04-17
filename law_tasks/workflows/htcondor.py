import os
from typing import List

import law
from law.contrib import htcondor
from law.util import rel_path

from law_tasks.workflows.common import (
    Config,
    add_workflow_settings_from_cfg,
    get_script_dir,
)
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("htcondor")


class HTCondorWorkflow(htcondor.HTCondorWorkflow):
    def htcondor_output_directory(self) -> law.LocalDirectoryTarget:  # type: ignore
        return law.LocalDirectoryTarget(
            os.path.join(get_script_dir(), "runs", "htcondor", self.__class__.__name__),
        )  # TODO Make dependent on law output

    def htcondor_bootstrap_file(self) -> law.JobInputFile:  # type: ignore
        bootstrap_file = rel_path(__file__, "bootstrap.sh")
        return law.JobInputFile(bootstrap_file, share=True, render_job=True)

    def htcondor_job_config(
        self,
        config: Config,
        job_num: int,
        branches: List[int],
    ):
        config = super().htcondor_job_config(config, job_num, branches)
        config = add_workflow_settings_from_cfg(self, config, workflow_type="htcondor")

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(get_script_dir(), "pyproject.toml"),
        )
        config.input_files["setup.sh"] = law.JobInputFile(
            os.path.join(get_script_dir(), "setup.sh"),
        )

        config.custom_content.append(("getenv", "true"))

        config.stdout = "stdout.txt"
        config.stderr = "stderr.txt"
        config.log = "condor.log"

        return config
