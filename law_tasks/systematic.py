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


class SystematicTask(HydraMixin, law.LocalWorkflow):
    """
    Systematic task that coordinates EnsembleTasks.
    
    Runs locally to coordinate ensemble branches. The actual training
    happens in FoldTasks which are submitted to HTCondor by EnsembleTask.
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

    def create_branch_map(self):
        """Define workflow branches - one per ensemble"""
        num_ensembles: int = self.estimator_config.expands.ensembles.num_ensembles or 1
        num_ensembles = max(1, num_ensembles)
        return {
            ensemble_index: {"ensemble": ensemble_index}
            for ensemble_index in range(num_ensembles)
        }

    def requires(self):
        """Workflow requires nothing; branches require EnsembleTask"""
        if self.is_branch():
            # Branch task: require the specific EnsembleTask
            ensemble_index = self.branch_data["ensemble"]
            return EnsembleTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=ensemble_index,
            )
        # Workflow container: no requirements
        return []

    def output(self) -> Dict[str, Any]:
        if self.is_branch():
            # Branch output: individual ensemble result
            os.makedirs(self.abs_results_path, exist_ok=True)
            return {
                "outputs": law.LocalDirectoryTarget(
                    f"{self.abs_results_path}/systematic_{self.estimator}_{self.systematic}_ens_{self.branch}.json"
                )
            }
        # Workflow output: collection of all branches
        return law.SiblingFileCollection(
            law.LocalDirectoryTarget(
                f"{self.abs_results_path}/systematic_{self.estimator}_{self.systematic}_ens_{{branch}}.json"
            )
        )

    def run(self):
        # Only branches run - workflow just coordinates
        if not self.is_branch():
            raise Exception("Workflow container should not run")
        
        # Branch task: collect result from its single EnsembleTask
        systematic_results = SystematicResults()
        
        ensemble_outputs = self.input()
        ensemble_result = EnsembleResults.from_json(ensemble_outputs["outputs"].path)
        systematic_results.ensembles.append(ensemble_result)

        systematic_results.to_json(self.output()["outputs"].path)
