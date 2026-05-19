"""EnsembleTask - Manages training for a single ensemble group.

This module defines the EnsembleTask which is responsible for:
- Creating FoldTask instances for each cross-validation fold
- Managing fold configuration and parameter expansion
- Aggregating results from all folds into an ensemble results object
- Handling ensemble-specific training configurations

The task forms the fourth level of the task DAG hierarchy:
    MainTask
    └── EstimatorTask (one per estimator)
         └── SystematicTask (one per systematic variation)
              └── EnsembleTask (one per ensemble group)
                   └── FoldTask (one per cross-validation fold)
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import law
import luigi

from needle.law_tasks.fold import FoldTask
from needle.law_tasks.mixins import HydraMixin
from needle.utils.config_schema import EstimatorConfig
from needle.utils.logging import ColorFormatter
from needle.utils.results import EnsembleResults, FoldResults

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(HydraMixin, law.Task):
    """Task representing a single ensemble training run.

    Creates FoldTask instances for each cross-validation fold and aggregates their results
    into an ensemble results object.
    """

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
        """Get the configuration for the associated estimator.

        Returns:
            EstimatorConfig: Configuration object for this ensemble's estimator.
        """
        return self.config.estimators[self.estimator]

    def requires(self):
        """Create FoldTask instances for all cross-validation folds.

        Returns:
            List[FoldTask]: One task per fold (0 to n_folds-1).
        """
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

    def input(self) -> List[Dict[str, law.LocalTarget]]:
        """Retrieve and flatten fold task outputs for local use.

        Unwraps remote fold outputs (which are nested in collection/jobs structure) to match
        the local output format, ensuring consistent access to fold results regardless of
        whether tasks ran locally or remotely.

        Returns:
            List[Dict[str, law.LocalTarget]]: Flattened fold outputs, one per fold.
        """
        _remote_fold_outputs = super().input()

        _flattened_fold_outputs = []

        for fold_output in _remote_fold_outputs:
            _flattened_fold_outputs.append(FoldTask.output_as_dict(fold_output))

        return _flattened_fold_outputs

    def output(self) -> Dict[str, Any]:
        """Define output target for aggregated ensemble results.

        Returns:
            Dict[str, Any]: Dictionary with 'outputs' file target containing ensemble results JSON.
        """
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("ensemble_results.json", type="f"),
        }

    def run(self) -> None:
        """Gather results from child FoldTask and merge them into own result container"""
        fold_results = [
            FoldResults.from_json(fold_output["outputs"].path) for fold_output in self.input()  # type: ignore
        ]
        EnsembleResults(folds=fold_results).to_json(self.output()["outputs"].path)
