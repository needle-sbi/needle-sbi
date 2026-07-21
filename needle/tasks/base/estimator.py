from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Type
from urllib.parse import urlencode

import luigi
from omegaconf import open_dict

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_schema import EstimatorConfig, SystematicConfig
from needle.utils.logging import ColorFormatter
from needle.utils.results import EstimatorResults, SystematicResults

logger = ColorFormatter.get_logger("estimator")


class BaseEstimatorTask(HydraParamsMixin, luigi.Task):
    """Backend-agnostic base for EstimatorTask."""

    results_path: str = luigi.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )  # type: ignore
    estimator: str = luigi.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore

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
                    key = urlencode(
                        {
                            "syst": systematic_task.systematic,
                            "ensem": ensemble_task.ensemble,
                            "fold": fold_task.fold_index,
                        }
                    )
                    model_paths_dict[key] = fold_task.output_as_dict(fold_task.output())["ckpt"].path  # type: ignore

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

    def output(self) -> Dict[str, Any]:
        base = str(self.abs_results_path)
        return {
            "outputs": luigi.LocalTarget(os.path.join(base, "estimator_result.json")),
            "input_models": luigi.LocalTarget(os.path.join(base, "input_models.json")),
        }

    def run(self) -> None:
        systematic_results = [
            SystematicResults.from_json(systematic_result["outputs"].path) for systematic_result in self.input()
        ]
        EstimatorResults(systematics=systematic_results).to_json(self.output()["outputs"].path)

        with open(self.output()["input_models"].path, "w") as f:
            json.dump(self.input_model_paths, f)
