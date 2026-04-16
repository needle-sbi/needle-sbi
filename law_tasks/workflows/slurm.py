import os
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

    def slurm_job_config(
        self,
        config: slurm.SlurmJobFileFactory.Config,
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
                    "The path to the root directory of the project should end with 'orchestrator' " f"but is {_script_dir}"
                )

            return str(_script_dir)

        config = super().slurm_job_config(config, job_num, branches)

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(get_script_dir(), "pyproject.toml"),
        )
        config.input_files["setup.sh"] = law.JobInputFile(
            os.path.join(get_script_dir(), "setup.sh"),
        )

        config.stdout = "stdout_%j.txt"  # %j = Slurm job id
        config.stderr = "stderr_%j.txt"

        return config
