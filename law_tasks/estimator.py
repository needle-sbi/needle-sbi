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


class EstimatorTask(law.htcondor.HTCondorWorkflow, law.Task, HydraMixin):
    """
    Estimator task that can run in two modes:
    - local: Creates and runs SystematicTasks in-process
    - htcondor: Submits each SystematicTask as a separate HTCondor job
    """
    rel_results_path: str = law.Parameter(
        description="Directory where the estimator results will be saved.",
        default="runs/estimator",
        significant=False,
    )  # type: ignore
    workflow = luigi.ChoiceParameter(
        default="local",
        choices=["local", "htcondor"],
        significant=False,
        description="Execution mode: local (run in-process) or htcondor (submit to cluster)"
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

    def requires(self) -> List[SystematicTask]:
        """
        Define task dependencies:
        - For workflow branches: Create the specific SystematicTask for this branch
        - For workflow container: No requirements (LAW handles it)
        """
        if self.is_branch():
            # This is a branch task - create the specific SystematicTask for this systematic
            systematic_key = self.branch_data["systematic"]
            return SystematicTask.req(
                self,
                config_file=self.config_file,
                workflow=self.workflow,
                estimator=self.estimator,
                systematic=systematic_key,
            )
        else:
            # This is the workflow container (branch=-1) - no direct requirements
            return []
    
    # ========================================================================
    # HTCondor Workflow Methods (only used when workflow="htcondor")
    # ========================================================================
    
    def create_branch_map(self):
        """Define workflow branches - single branch per systematic"""
        # Always return branch map since we inherit from HTCondorWorkflow
        # In local mode, workflow machinery still runs but doesn't submit to HTCondor
        return {
            idx: {"systematic": syst_key}
            for idx, syst_key in enumerate(self.estimator_config.expands.systematics.keys())
        }
    
    def workflow_requires(self):
        """Workflow-level requirements"""
        return {}
    
    def htcondor_output_directory(self):
        """Directory for HTCondor job logs"""
        return law.LocalDirectoryTarget(
            f".law/htcondor/{self.task_family}/{self.estimator}"
        )
    
    def htcondor_bootstrap_file(self):
        """Bootstrap script for remote environment setup"""
        return law.util.rel_path(__file__, "../../setup.sh")
    
    def htcondor_job_config(self, config, job_num, branches):
        """Configure HTCondor resources for SystematicTasks"""
        # Only applies when workflow="htcondor" (LAW handles mode switching)
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
        return config

    def output(self) -> Dict[str, Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        return {"outputs": law.LocalFileTarget(f"{self.abs_results_path}/estimator_outputs.json")}

    def run(self):
        systematic_results = EstimatorResults()

        for fold_outputs in self.input():
            fold_result = SystematicResults.from_json(fold_outputs["outputs"].path)
            systematic_results.systematics.append(fold_result)

        systematic_results.to_json(self.output()["outputs"].path)
