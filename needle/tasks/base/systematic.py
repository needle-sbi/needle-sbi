from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Type

import luigi

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_schema import EstimatorConfig, SystematicConfig
from needle.utils.logging import ColorFormatter
from needle.utils.results import EnsembleResults, SystematicResults

logger = ColorFormatter.get_logger("systematic")


class BaseSystematicTask(HydraParamsMixin, luigi.Task):
    """Backend-agnostic base for SystematicTask."""

    results_path: str = luigi.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )  # type: ignore
    estimator: str = luigi.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore
    systematic: str = luigi.Parameter(
        description="Name of the systematic uncertainty.",
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return Path(
            os.path.join(
                os.path.abspath(self.results_path),
                f"est__{self.estimator}",
                f"syst__{self.systematic}",
            )
        )

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        return self.config.estimators[self.estimator].expands.systematics[self.systematic]

    def _ensemble_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _ensemble_task_class()")

    def requires(self) -> List[Any]:
        EnsembleTask = self._ensemble_task_class()
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
        return {
            "outputs": luigi.LocalTarget(
                os.path.join(str(self.abs_results_path), "systematic_results.json")
            )
        }

    def run(self) -> None:
        ensemble_results = [
            EnsembleResults.from_json(ensemble_result["outputs"].path) for ensemble_result in self.input()
        ]
        SystematicResults(ensemble_results).to_json(self.output()["outputs"].path)
