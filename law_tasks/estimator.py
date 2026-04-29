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


class EstimatorTask(HydraMixin, law.Task):
    """
    Estimator task that coordinates SystematicTasks.
    
    Plain task (not a workflow) that requires all SystematicTasks for this estimator.
    This allows SystematicTasks to run in parallel, and they trigger EnsembleTask workflows.
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
    
    def requires(self):
        """Require all SystematicTasks for this estimator"""
        return [
            SystematicTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=systematic_key,
                use_htcondor=self.use_htcondor,
            )
            for systematic_key in self.estimator_config.expands.systematics.keys()
        ]
    
    def output(self) -> Dict[str, Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        return law.LocalFileTarget(
            f"{self.abs_results_path}/estimator_{self.estimator}.json"
        )

    def run(self):
        """Collect results from all SystematicTasks"""
        estimator_results = EstimatorResults()
        
        # Input is a list of SystematicTask outputs
        for systematic_output in self.input():
            systematic_result = SystematicResults.from_json(systematic_output.path)
            estimator_results.systematics.append(systematic_result)

        estimator_results.to_json(self.output().path)
