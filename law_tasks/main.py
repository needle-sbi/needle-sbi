import os
from pathlib import Path
from typing import List

import law

from law_tasks.estimator import EstimatorTask
from law_tasks.mixins import HydraMixin


class MainTask(HydraMixin, law.WrapperTask):
    """Root Task that is the main entry point for the Task Graph

    Will run all EstimatorTasks listed in the config.
    """

    results_path: str = law.Parameter(
        description="Directory where the training results will be saved.",
        default="runs",
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.results_path)  # type: ignore

    def requires(self) -> List[EstimatorTask]:
        return [
            EstimatorTask(
                config_file=self.config_file,
                estimator=estimator_key,
                results_path=os.path.join(self.abs_results_path, f"est__{estimator_key}"),
            )
            for estimator_key in self.config.estimators.keys()
        ]
