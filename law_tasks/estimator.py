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


class EstimatorTask(HydraMixin, law.htcondor.HTCondorWorkflow, law.LocalWorkflow):
    """
    Estimator task that runs SystematicTasks.    
    By default runs as HTCondor workflow (submits branches to cluster).
    Use --EstimatorTask-workflow local to run branches locally.    """
    rel_results_path: str = law.Parameter(
        description="Directory where the estimator results will be saved.",
        default="runs/estimator",
        significant=False,
    )  # type: ignore
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore

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
                systematic=systematic_key,
            )
        # Workflow container: no requirements
        return []

    def htcondor_output_directory(self):
        """Directory for HTCondor job logs"""
        return law.LocalDirectoryTarget(
            os.path.join(os.getcwd(), ".law", "htcondor", self.task_family, self.estimator)
        )
    
    def htcondor_bootstrap_file(self):
        """Bootstrap script for remote environment setup"""
        # Get absolute path to setup.sh in workspace root
        task_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(task_dir)
        return os.path.join(workspace_root, "setup.sh")
    
    def htcondor_job_config(self, config, job_num, branches):
        """Configure HTCondor resources"""
        config.custom_content.append(("request_cpus", "2"))
        config.custom_content.append(("request_memory", "16000"))
        config.custom_content.append(("request_runtime", "7200"))
        config.custom_content.append(("request_GPUs", "0"))
        config.custom_content.append(("getenv", "True"))
        config.custom_content.append(("universe", "vanilla"))
        config.custom_content.append((
            "+Environment",
            '"FAIR_UNIVERSE_DATA=/data/dust/group/atlas/needle/FAIRUnv/'
            'UncertaintyChallenge_2024/ProcessedData_v1_2025-10-03/CombData-part0.parquet"'
        ))
        # Configure log transfer - capture stdout/stderr
        config.custom_content.append(("should_transfer_files", "YES"))
        config.custom_content.append(("when_to_transfer_output", "ON_EXIT"))
        config.custom_content.append(("transfer_output_files", ""))
        return config
    
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
