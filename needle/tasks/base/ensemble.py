from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Type

import luigi

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_schema import EstimatorConfig
from needle.utils.logging import ColorFormatter
from needle.utils.results import EnsembleResults, FoldResults

logger = ColorFormatter.get_logger("ensemble")


class BaseEnsembleTask(HydraParamsMixin, luigi.Task):
    """Backend-agnostic base for EnsembleTask."""

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
    ensemble: int = luigi.IntParameter(
        description="Index of the ensemble (type: int).",
        default=0,
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return Path(
            os.path.join(
                os.path.abspath(self.results_path),
                f"est__{self.estimator}",
                f"syst__{self.systematic}",
                f"ensem__{self.ensemble}",
            )
        )

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def _fold_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _fold_task_class()")

    def requires(self) -> List[Any]:
        FoldTask = self._fold_task_class()
        return [
            FoldTask(
                config_file=self.config_file,
                hydra_overrides=self.hydra_overrides,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=self.ensemble,
                fold_index=fold_index,
                results_path=self.results_path,
            )
            for fold_index in range(self.estimator_config.expands.folds)
        ]

    def output(self) -> Dict[str, Any]:
        return {
            "outputs": luigi.LocalTarget(
                os.path.join(str(self.abs_results_path), "ensemble_results.json")
            )
        }

    def run(self) -> None:
        fold_results = [
            FoldResults.from_json(fold_output["outputs"].path) for fold_output in self.input()
        ]
        EnsembleResults(folds=fold_results).to_json(self.output()["outputs"].path)
