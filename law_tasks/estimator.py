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


class EstimatorTask(HydraMixin, law.Task):
    results_path: str = law.Parameter(
        description="Directory where the estimator results will be saved.",
        significant=False,
    )  # type: ignore
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.results_path)  # type: ignore

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def requires(self) -> List[SystematicTask]:
        return [
            SystematicTask(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=systematic_key,
                results_path=os.path.join(self.abs_results_path, f"syst__{systematic_key}"),
            )
            for systematic_key in self.estimator_config.expands.systematics.keys()
        ]

    def output(self) -> Dict[str, Any]:
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("estimator_result.json", type="f"),
        }

    def run(self):
        """Gather results from all SystematicTasks and merge them into own container"""
        systematic_results = [
            SystematicResults.from_json(systematic_result["outputs"].path) for systematic_result in self.input()
        ]
        EstimatorResults(systematics=systematic_results).to_json(self.output()["outputs"].path)
