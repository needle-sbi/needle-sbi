import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode

import law
from omegaconf import open_dict

from law_tasks.fold import FoldTask
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
        systematics = est.expands.systematics or {}

        with open_dict(est):  # type: ignore
            """Account for different scenarios when using 'nominal' systematics case.

            The checks to perform are:
            1. Are there other systematics than just the nominal case?
            2. Is the nominal case still in the Dict?
            3. Is the nominal case just the default? -> Remove if others were provided

            Note:
                A more robust approach would be to completely rework the way the Law Tasks are built, bypassing
                for example the SystematicsTask is no systematics are provided.
            """
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
                    model_paths_dict[key] = FoldTask.output_as_dict(fold_task.output())["ckpt"].path  # type: ignore

        return model_paths_dict

    def requires(self) -> List[SystematicTask]:
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

        with open(self.output()["input_models"].path, "w") as f:
            json.dump(self.input_model_paths, f)
