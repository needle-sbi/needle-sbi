import os
from pathlib import Path
from typing import Any, Dict, List

import law
import luigi

# Load HTCondor contrib for workflow support
law.contrib.load("htcondor")

from law_tasks.mixins import HydraMixin
from law_tasks.systematic import SystematicTask
from orchestrator.config import EstimatorConfig
from orchestrator.results import EstimatorResults, SystematicResults
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("estimator")


class EstimatorTask(HydraMixin, law.LocalWorkflow):
    """
    Estimator task that coordinates SystematicTasks.
    
    Runs locally to coordinate systematic branches. The actual training
    happens in FoldTasks which are submitted to HTCondor by EnsembleTask.
    """
    rel_results_path: str = law.Parameter(
        description="Directory where the estimator results will be saved.",
        default="runs/estimator",
        significant=False,
    )  # type: ignore
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
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
        estimator_file = Path(__file__)  # law_tasks/estimator.py
        law_tasks_dir = estimator_file.parent  # law_tasks/
        workspace_root = law_tasks_dir.parent  # orchestrator/
        
        return (workspace_root / results_path).resolve()

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def create_branch_map(self):
        """Define workflow branches - one per systematic"""
        return {
            idx: {"systematic": syst_key}
            for idx, syst_key in enumerate(self.estimator_config.expands.systematics.keys())
        }
    
    def requires(self):
        """Workflow requires nothing; branches require SystematicTask"""
        if self.is_branch():
            # Branch task: require the specific SystematicTask
            systematic_key = self.branch_data["systematic"]
            return SystematicTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=systematic_key,                use_htcondor=self.use_htcondor,            )
        # Workflow container: no requirements
        return []
    
    def output(self) -> Dict[str, Any]:
        if self.is_branch():
            # Branch output: individual systematic result
            os.makedirs(self.abs_results_path, exist_ok=True)
            return {
                "outputs": law.LocalFileTarget(
                    f"{self.abs_results_path}/estimator_{self.estimator}_syst_{self.branch}.json"
                )
            }
        # Workflow output: collection of all branches
        return law.SiblingFileCollection(
            law.LocalFileTarget(
                f"{self.abs_results_path}/estimator_{self.estimator}_syst_{{branch}}.json"
            )
        )

    def run(self):
        # Only branches run - workflow just coordinates
        if not self.is_branch():
            raise Exception("Workflow container should not run")
        
        # Branch task: collect results from its single SystematicTask
        systematic_results = EstimatorResults()
        
        fold_outputs = self.input()
        fold_result = SystematicResults.from_json(fold_outputs["outputs"].path)
        systematic_results.systematics.append(fold_result)

        systematic_results.to_json(self.output()["outputs"].path)
