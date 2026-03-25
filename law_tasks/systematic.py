"""
Task for a single systematic uncertainty
"""
import os
from pathlib import Path
from typing import Any, Dict

import law

from law_tasks.ensemble import EnsembleTask
from law_tasks.mixins import HydraMixin
from orchestrator.config import EstimatorConfig, SystematicConfig
from orchestrator.results import EnsembleResults, SystematicResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("systematic")


class SystematicTask(HydraMixin, law.Task):
    results_path = law.Parameter(
        description="Directory where the systematic results will be saved.",
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
        return os.path.abspath(self.results_path)  # type: ignore

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        return self.config.estimators[self.estimator].expands.systematics[self.systematic]

    def requires(self):
        num_ensembles: int = self.estimator_config.expands.ensembles.num_ensembles or 1
        num_ensembles = max(1, num_ensembles)

        return [
            EnsembleTask(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=ensemble_index,
                results_path=os.path.join(self.abs_results_path, f"ensem__{ensemble_index}"),
            )
            for ensemble_index in range(num_ensembles)
        ]

    def output(self) -> Dict[str, Any]:
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("systematic_results.json", type="f"),
        }

    def run(self):
        ensemble_results = [
            EnsembleResults.from_json(ensemble_result["outputs"].path) for ensemble_result in self.input()
        ]
        SystematicResults(ensemble_results).to_json(self.output()["outputs"].path)
