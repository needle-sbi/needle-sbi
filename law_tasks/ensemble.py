"""
Task for a single ensemble training, includes multiple folds.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import law
import luigi

from law_tasks.fold import FoldTask
from law_tasks.mixins import HydraMixin
from orchestrator.config import EstimatorConfig
from orchestrator.results import EnsembleResults, FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("ensemble")

FoldTaskOutput = List[Dict[str, law.LocalFileTarget] | Dict[str, law.TargetCollection]]


class EnsembleTask(HydraMixin, law.Task):
    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )  # type: ignore
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore
    systematic: str = law.Parameter(
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

    def requires(self):
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
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("ensemble_results.json", type="f"),
        }

    def input(self) -> List[Dict[str, law.LocalTarget]]:
        """Unpack local and remote inputs

         1. Local is simply the Dict defined in the output method of the FoldTask
         2. Remote is instead a DotDict with 'collection' and 'jobs' fields.

        Example:
            print(super().input())
            [
                DotDict(
                    {
                        "jobs": law.LocalFileTarget(),
                        "collection": law.TargetCollection(len=1)
                    }
                )
            ]

        Returns:
            Dict[str, str]: Properly formatted output Dict with key:Target pairs
        """
        _flattened_fold_outputs = []
        fold_inputs: FoldTaskOutput = super().input()

        for fold_output in fold_inputs:
            remote_collection: law.TargetCollection | None = fold_output.get("collection")  # type: ignore

            if remote_collection:
                if len(remote_collection) != 1:
                    raise NotImplementedError(
                        "Currently the usage of branches in FoldTask is not supported. Instead, folds "
                        "have to be their own Task instance required by EnsembleTask."
                    )
                _flattened_fold_outputs.append(remote_collection[0])
            else:
                _flattened_fold_outputs.append(fold_output)

        return _flattened_fold_outputs

    def run(self) -> None:
        """Gather results from child FoldTask and merge them into own result container"""
        fold_results = [
            FoldResults.from_json(fold_output["outputs"].path) for fold_output in self.input()  # type: ignore
        ]
        EnsembleResults(folds=fold_results).to_json(self.output()["outputs"].path)
