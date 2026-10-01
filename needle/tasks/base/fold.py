from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Type

import luigi

from needle.tasks.base.expansion import BaseExpansionTask
from needle.utils.config_schema import EstimatorConfig, SystematicConfig
from needle.utils.config_utils import merge_systematic_config


class BaseFoldTask(BaseExpansionTask):
    """Fan-out wrapper for a single cross-validation fold.

    Requires one ``TrainingTask`` and signals completion via a ``.done`` marker.
    Backends override ``_training_task_class()`` to inject the appropriate leaf task.
    """

    systematic: str = luigi.Parameter(
        description="Name of the systematic uncertainty.",
        default="nominal",
        significant=True,
    )  # type: ignore
    ensemble: int = luigi.IntParameter(
        description="Index of the ensemble (type: int).",
        default=0,
        significant=True,
    )  # type: ignore
    fold_index: int = luigi.IntParameter(
        description="Index of the cross-validation fold (type: int)",
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
                f"fold__{self.fold_index}",
            )
        )

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        return merge_systematic_config(self.estimator_config, self.systematic)

    def _training_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _training_task_class()")

    def requires(self) -> List[Any]:
        TrainingTask = self._training_task_class()
        return [
            TrainingTask(
                config_file=str(self.config_file),
                hydra_overrides=self.hydra_overrides,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=self.ensemble,
                fold_index=self.fold_index,
                results_path=self.results_path,
            )
        ]
