from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Type

import luigi

from needle.tasks.base.expansion import BaseExpansionTask
from needle.utils.config_schema import EstimatorConfig


class BaseEnsembleTask(BaseExpansionTask):
    """Fan-out wrapper for a single ensemble group.

    Requires one ``FoldTask`` per fold and signals completion via a ``.done`` marker.
    Backends override ``_fold_task_class()`` to inject the appropriate FoldTask variant.
    """

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
