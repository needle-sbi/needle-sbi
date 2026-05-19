import os
from pathlib import Path
from typing import Any, Dict, List

import law

from needle.law_tasks.ensemble import EnsembleTask
from needle.law_tasks.mixins import HydraMixin
from needle.utils.config_schema import EstimatorConfig, SystematicConfig
from needle.utils.logging import ColorFormatter
from needle.utils.results import EnsembleResults, SystematicResults

logger = ColorFormatter.get_logger("systematic")


class SystematicTask(HydraMixin, law.Task):
    """Task representing a systematic uncertainty variation.

    Creates `EnsembleTask` instances for multiple ensemble runs of the same systematic.
    Aggregates results from all ensembles into a single systematic results object.
    """

    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )  # type: ignore
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
        return Path(
            os.path.join(os.path.abspath(self.results_path), f"est__{self.estimator}", f"syst__{self.systematic}")
        )

    @property
    def estimator_config(self) -> EstimatorConfig:
        """Get the configuration for the associated parent estimator.

        Returns:
            EstimatorConfig
        """
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        """Get the configuration for this specific systematic uncertainty.

        Returns:
            SystematicConfig
        """
        return self.config.estimators[self.estimator].expands.systematics[self.systematic]

    def requires(self) -> List[EnsembleTask]:
        """Create `EnsembleTask` instances for all ensemble runs.

        Returns:
            List[EnsembleTask]: One task per ensemble index (0 to num_ensembles-1).
        """
        num_ensembles: int = self.estimator_config.expands.ensembles.num_ensembles or 1
        num_ensembles = max(1, num_ensembles)

        return [
            EnsembleTask(
                config_file=str(self.config_file),
                hydra_overrides=self.hydra_overrides,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=ensemble_index,
                results_path=self.results_path,
            )
            for ensemble_index in range(num_ensembles)
        ]

    def output(self) -> Dict[str, Any]:
        """Define output target for aggregated systematic results.

        Returns:
            Dict[str, Any]: Dictionary with 'outputs' file target containing systematic results JSON.
        """
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("systematic_results.json", type="f"),
        }

    def run(self) -> None:
        """Aggregate the results from all Ensembles and store them in a JSON"""
        ensemble_results = [
            EnsembleResults.from_json(ensemble_result["outputs"].path) for ensemble_result in self.input()
        ]
        SystematicResults(ensemble_results).to_json(self.output()["outputs"].path)
