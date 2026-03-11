import os
from pathlib import Path
from typing import Any, Dict, List

import law

from law_tasks.mixins import HydraMixin
from law_tasks.systematic import SystematicTask
from orchestrator.config import EstimatorConfig
from orchestrator.results import EstimatorResults, SystematicResults
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("estimator")


class EstimatorTask(law.Task, HydraMixin):
    rel_results_path: str = law.Parameter(
        description="Directory where the estimator results will be saved.",
        default="runs/estimator",
        significant=False,
    )  # type: ignore
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def requires(self) -> List[SystematicTask]:
        nominal = [
            SystematicTask.req(
                self,
                config_file=self.config_file,
                systematic="nominal",
            )
        ]
        systematic_shifts = [
            SystematicTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=systematic_key,
            )
            for systematic_key in self.estimator_config.expands.systematics.keys()
        ]
        return nominal + systematic_shifts

    def output(self) -> Dict[str, Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        return {"outputs": law.LocalFileTarget(f"{self.abs_results_path}/estimator_outputs.json")}

    def run(self):
        systematic_results = EstimatorResults()

        for fold_outputs in self.input():
            fold_result = SystematicResults.from_json(fold_outputs["outputs"].path)
            systematic_results.systematics.append(fold_result)

        systematic_results.to_json(self.output()["outputs"].path)
