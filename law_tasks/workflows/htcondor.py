import os
from typing import List

import law
from law.contrib import htcondor
from law.util import rel_path


class HTCondorWorkflow(htcondor.HTCondorWorkflow):
    def htcondor_output_directory(self) -> law.LocalDirectoryTarget:  # type: ignore
        return law.LocalDirectoryTarget(os.path.join("/tmp/law_output", "htcondor", self.__class__.__name__))

    def htcondor_bootstrap_file(self) -> law.JobInputFile:  # type: ignore
        bootstrap_file = rel_path(__file__, "bootstrap.sh")
        return law.JobInputFile(bootstrap_file, share=True, render_job=True)

    def htcondor_job_config(
        self,
        config: htcondor.HTCondorJobFileFactory.Config,
        job_num: int,
        branches: List[int],
    ):
        script_dir = os.getenv("SCRIPT_DIR")

        if not script_dir:
            raise ValueError("Variable 'SCRIPT_DIR' is not defined")

        config = super().htcondor_job_config(config, job_num, branches)

        config.render_variables["SCRIPT_DIR"] = script_dir

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(script_dir, "pyproject.toml"),
        )
        config.input_files["setup.sh"] = law.JobInputFile(
            os.path.join(script_dir, "setup.sh"),
        )

        config.custom_content.append(("getenv", "true"))

        config.stdout = "stdout.txt"
        config.stderr = "stderr.txt"
        config.log = "condor.log"

        return config
