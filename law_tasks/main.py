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
        description="Root directory where results are saved.",
        default="runs",
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        if self.config.results_path:
            self.results_path = self.config.results_path

        return os.path.abspath(self.results_path)  # type: ignore

    def requires(self) -> List[EstimatorTask]:
        return [
            EstimatorTask(
                config_file=self.config_file,
                hydra_overrides=self.hydra_overrides,
                estimator=estimator_key,
                results_path=self.abs_results_path,
            )
            for estimator_key in self.config.estimators.keys()
        ]
