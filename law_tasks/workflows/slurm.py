import os
from pathlib import Path
from pathlib import Path
from typing import List

import law
from law.contrib import slurm
from law.util import rel_path


class SlurmWorkflow(slurm.SlurmWorkflow):
    def slurm_output_directory(self) -> law.LocalDirectoryTarget:  # type: ignore
        return law.LocalDirectoryTarget(os.path.join("/tmp/law_output", "slurm", self.__class__.__name__))

    def slurm_bootstrap_file(self):  # type: ignore
        bootstrap_file = os.path.join(rel_path(__file__), "bootstrap.sh")
        return law.JobInputFile(bootstrap_file, share=True, render_job=True)

    @property
    def script_dir(self) -> str:
        _script_dir = os.getenv("SCRIPT_DIR")

        if not _script_dir:
            _script_dir = Path(os.path.abspath(__file__)).parent.parent.parent

        return str(_script_dir)

    def slurm_job_config(
        self,
        config: slurm.SlurmJobFileFactory.Config,
        job_num: int,
        branches: List[int],
    ):
        config = super().slurm_job_config(config, job_num, branches)

        config.render_variables["SCRIPT_DIR"] = self.script_dir

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(self.script_dir, "pyproject.toml"),
        )
        config.input_files["setup.sh"] = law.JobInputFile(
            os.path.join(self.script_dir, "setup.sh"),
        )

        config.stdout = "stdout_%j.txt"  # %j = Slurm job id
        config.stderr = "stderr_%j.txt"

        return config
