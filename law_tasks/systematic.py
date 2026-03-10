"""
Task for a single systematic uncertainty
"""
import os
from pathlib import Path
from typing import Any, Dict

import law

from law_tasks.ensemble import EnsembleTask
from law_tasks.mixins import HydraMixin
from orchestrator.config import SystematicConfig, EstimatorConfig
from orchestrator.results import EnsembleResults, SystematicResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("systematic")


class SystematicTask(law.Task, HydraMixin):
    rel_results_path = law.Parameter(
        description="Directory where the systematic results will be saved.",
        default="runs/systematic",  # TODO
        significant=False,
    )
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore
    systematic: str = law.Parameter(
        description="Name of the systematic uncertainty.",
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore
    
    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]
    
    @property
    def systematic_config(self) -> SystematicConfig:
        return self.config.estimators[self.estimator].expands.systematics[self.systematic]

    def requires(self):
        return [
            EnsembleTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=ensemble_index,
            )
            for ensemble_index in range(self.estimator_config.expands.ensembles.num_ensembles)
        ]
    
    def output(self) -> Dict[str, Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        return {"outputs": law.LocalDirectoryTarget(f"{self.abs_results_path}/systematic_results.json")}

    def run(self):
        systematic_results = SystematicResults()

        for ensemble_output in self.input():
            ensemble_result = EnsembleResults.from_json(ensemble_output["outputs"].path)
            systematic_results.ensembles.append(ensemble_result)

        systematic_results.to_json(self.output()["outputs"].path)
