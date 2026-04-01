import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode

import law
from omegaconf import open_dict

from law_tasks.mixins import HydraMixin
from law_tasks.systematic import SystematicTask
from orchestrator.config import EstimatorConfig, SystematicConfig
from orchestrator.results import EstimatorResults, SystematicResults
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("estimator")


class EstimatorTask(HydraMixin, law.Task):
    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )  # type: ignore
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return Path(os.path.join(os.path.abspath(self.results_path), f"est__{self.estimator}"))

    @property
    def estimator_config(self) -> EstimatorConfig:
        est = self.config.estimators[self.estimator]

        with open_dict(est):  # type: ignore
            if not est.expands.systematics:
                est.expands.systematics["nominal"] = SystematicConfig()

        return est

    def record_model_paths(self) -> None:
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
                    path = fold_task.output()["ckpt"].path
                    model_paths_dict[key] = path

        with open(self.output()["input_models"].path, "w") as f:
            json.dump(model_paths_dict, f)

    def requires(self) -> List[SystematicTask]:
        return [
            SystematicTask(
                config_file=str(self.config_file),
                estimator=self.estimator,
                systematic=systematic_key,
                results_path=self.results_path,
            )
            for systematic_key in self.estimator_config.expands.systematics.keys()
        ]

    def output(self) -> Dict[str, Any]:
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("estimator_result.json", type="f"),
            "input_models": base.child("input_models.json", type="f"),
        }

    def run(self) -> None:
        """Gather results from all SystematicTasks and merge them into own container"""
        systematic_results = [
            SystematicResults.from_json(systematic_result["outputs"].path) for systematic_result in self.input()
        ]
        EstimatorResults(systematics=systematic_results).to_json(self.output()["outputs"].path)
        self.record_model_paths()
