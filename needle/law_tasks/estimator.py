import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode

import law
from omegaconf import open_dict

from needle.law_tasks.fold import FoldTask
from needle.law_tasks.mixins import HydraMixin
from needle.law_tasks.systematic import SystematicTask
from needle.utils.config_schema import EstimatorConfig, SystematicConfig
from needle.utils.logging import ColorFormatter
from needle.utils.results import EstimatorResults, SystematicResults

logger = ColorFormatter.get_logger("estimator")


class EstimatorTask(HydraMixin, law.Task):
    """Task representing a single estimator. An estimator is a similar concept to a single type
    of model used in your analysis chain. Since `SystematicTask` also has models of the same type but
    with a different meaning, we wanted to differentiate between `EstimatorTask` (a model potentially
    trained with systematics) and a regular ML model.

    Creates `SystematicTask` instances for each systematic shift defined in the config. By default, this
    will be a single systematic called 'nominal' which just passes the information along to the next
    layer in the tree. If you wish to train on different systematics, each has to be registered in the
    config in the `expands` section of the estimator. See the corresponding documentation for more
    information.

    Handles systematic configuration merging and aggregates results from all systematic variations.
    """

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
        """Get the estimator configuration with normalized systematic cases.

        Handles the 'nominal' systematic special case:

         - If only 'nominal' exists and is the default (unmodified), keeps it
         - If other systematics exist alongside a default 'nominal', removes the nominal
         - Otherwise, preserves the systematic configuration as-is

        This normalization prevents redundant nominal case execution when other variations exist.

        Returns:
            EstimatorConfig: Estimator config with processed systematics dictionary.
        """
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
        """Collect checkpoint paths from all trained folds across systematics and ensembles.

        Traverses the full task hierarchy (systematic → ensemble → fold) to gather all model
        checkpoint paths. Keys encode the systematic/ensemble/fold indices for reference.

        Returns:
            Dict[str, str]: Mapping of URL-encoded task indices to checkpoint paths.
                Key format: `"syst=<name>&ensem=<idx>&fold=<idx>"`
        """
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
        """Create SystematicTask instances for each systematic uncertainty.

        Returns:
            List[SystematicTask]: One task per systematic key in the estimator config.
        """
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
        """Define output targets for aggregated estimator results.

        Returns:
            Dict[str, Any]: Dictionary with 'outputs' and 'input_models' file targets.
        """
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("estimator_result.json", type="f"),
            "input_models": base.child("input_models.json", type="f"),
        }

    def run(self) -> None:
        """Gather results from all `SystematicTasks` and merge them into own results container"""
        systematic_results = [
            SystematicResults.from_json(systematic_result["outputs"].path) for systematic_result in self.input()
        ]
        EstimatorResults(systematics=systematic_results).to_json(self.output()["outputs"].path)

        with open(self.output()["input_models"].path, "w") as f:
            json.dump(self.input_model_paths, f)
