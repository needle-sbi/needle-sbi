from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Type

import luigi

from needle.tasks.base.expansion import BaseExpansionTask
from needle.utils.config_schema import EstimatorConfig, SystematicConfig


class BaseSystematicTask(BaseExpansionTask):
    """Fan-out wrapper for a single systematic variation.

    Requires one ``EnsembleTask`` per ensemble and signals completion via a ``.done`` marker.
    Backends override ``_ensemble_task_class()`` to inject the appropriate EnsembleTask variant.
    """

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
        num_ensembles: int = max(1, self.estimator_config.expands.ensembles.num_ensembles or 1)
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
