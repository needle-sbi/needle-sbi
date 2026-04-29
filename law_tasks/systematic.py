"""
Task for a single systematic uncertainty
"""
import os
from pathlib import Path
from typing import Any, Dict

import law
import luigi

# Load HTCondor contrib for workflow support
law.contrib.load("htcondor")

from law_tasks.ensemble import EnsembleTask
from law_tasks.mixins import HydraMixin
from orchestrator.config import EstimatorConfig, SystematicConfig
from orchestrator.results import EnsembleResults, SystematicResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("systematic")


class SystematicTask(HydraMixin, law.Task):
    """
    Systematic task that coordinates EnsembleTasks.
    
    Plain task (not a workflow) that requires all EnsembleTasks for this systematic.
    This allows EnsembleTasks to run in parallel and submit to HTCondor.
    """
    rel_results_path = law.Parameter(
        description="Directory where the systematic results will be saved.",
        default="runs/systematic",
        significant=False,
    )
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore
    systematic: str = law.Parameter(
        description="Name of the systematic uncertainty.",
        significant=True,
    )  # type: ignore
    use_htcondor: bool = law.Parameter(
        description="Whether to use HTCondor for EnsembleTasks (passed down the chain).",
        default=False,
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        """Get absolute path to results directory.
        
        Uses __file__ to find workspace root so it works correctly on HTCondor
        worker nodes where cwd is the scratch directory.
        """
        results_path = Path(self.rel_results_path)
        
        if results_path.is_absolute():
            return results_path
        
        # Resolve relative to workspace root using module location
        systematic_file = Path(__file__)  # law_tasks/systematic.py
        law_tasks_dir = systematic_file.parent  # law_tasks/
        workspace_root = law_tasks_dir.parent  # orchestrator/
        
        return (workspace_root / results_path).resolve()

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        return self.config.estimators[self.estimator].expands.systematics[self.systematic]

    def requires(self):
        """Require all EnsembleTasks for this systematic"""
        num_ensembles: int = self.estimator_config.expands.ensembles.num_ensembles or 1
        num_ensembles = max(1, num_ensembles)
        return [
            EnsembleTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=ensemble_index,
                workflow="htcondor" if self.use_htcondor else "local",
            )
            for ensemble_index in range(num_ensembles)
        ]

    def output(self) -> Dict[str, Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        return law.LocalFileTarget(
            f"{self.abs_results_path}/systematic_{self.estimator}_{self.systematic}.json"
        )

    def run(self):
        """Collect results from all EnsembleTasks"""
        systematic_results = SystematicResults()
        
        # Input is a list of EnsembleTask outputs
        for ensemble_output in self.input():
            ensemble_result = EnsembleResults.from_json(ensemble_output.path)
            systematic_results.ensembles.append(ensemble_result)

        systematic_results.to_json(self.output().path)
