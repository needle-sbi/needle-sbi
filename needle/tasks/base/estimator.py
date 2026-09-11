from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Type
from urllib.parse import urlencode

import luigi
from omegaconf import open_dict

from needle.tasks.base.expansion import BaseExpansionTask
from needle.utils.config_schema import EstimatorConfig, SystematicConfig
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("estimator")


class BaseEstimatorTask(BaseExpansionTask):
    """Fan-out wrapper for a single estimator across all its systematic variations.

    Requires one ``SystematicTask`` per systematic key and signals completion via a ``.done``
    marker. Backends override ``_systematic_task_class()`` to inject the appropriate variant.
    """

    @property
    def abs_results_path(self) -> Path:
        return Path(os.path.join(os.path.abspath(self.results_path), f"est__{self.estimator}"))

    @property
    def estimator_config(self) -> EstimatorConfig:
        est = self.config.estimators[self.estimator]
        systematics = est.expands.systematics or {}

        with open_dict(est):  # type: ignore
            others_than_nominal = {k: v for k, v in systematics.items() if k != "nominal"}
            nominal_val = systematics.get("nominal") or SystematicConfig()
            nominal_is_default: bool = nominal_val == SystematicConfig()

            if others_than_nominal and nominal_is_default:
                systematics = others_than_nominal
            elif not others_than_nominal and nominal_is_default:
                systematics = {"nominal": nominal_val}

            est.expands.systematics = systematics

        return est

    @property
    def input_model_paths(self) -> Dict[str, str]:
        """Collect checkpoint paths from all trained folds across systematics and ensembles."""
        model_paths_dict: Dict[str, str] = {}

        for systematic_task in self.requires():
            for ensemble_task in systematic_task.requires():
                for fold_task in ensemble_task.requires():
                    (training_task,) = fold_task.requires()
                    key = urlencode(
                        {
                            "syst": systematic_task.systematic,
                            "ensem": ensemble_task.ensemble,
                            "fold": fold_task.fold_index,
                        }
                    )
                    model_paths_dict[key] = training_task.output_as_dict(training_task.output())["ckpt"].path  # type: ignore

        return model_paths_dict

    def _systematic_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _systematic_task_class()")

    def requires(self) -> List[Any]:
        SystematicTask = self._systematic_task_class()
        return [
            SystematicTask(
                config_file=str(self.config_file),
                hydra_overrides=self.hydra_overrides,
                estimator=self.estimator,
                systematic=systematic_key,
                results_path=self.results_path,
            )
            for systematic_key in self.estimator_config.expands.systematics.keys()
        ]
