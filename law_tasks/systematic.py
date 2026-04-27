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


class SystematicTask(law.htcondor.HTCondorWorkflow, law.Task, HydraMixin):
    """
    Systematic task that can run in two modes:
    - local: Creates and runs EnsembleTasks in-process
    - htcondor: Submits each EnsembleTask as a separate HTCondor job
    """
    rel_results_path = law.Parameter(
        description="Directory where the systematic results will be saved.",
        default="runs/systematic",  # TODO
        significant=False,
    )
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
    systematic: str = law.Parameter(
        description="Name of the systematic uncertainty.",
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        return self.config.estimators[self.estimator].expands.systematics[self.systematic]

    def requires(self):
        """
        Define task dependencies:
        - For workflow branches: Create the specific EnsembleTask for this branch
        - For workflow container: No requirements (LAW handles it)
        """
        if self.is_branch():
            # This is a branch task - create the specific EnsembleTask for this ensemble
            ensemble_index = self.branch_data["ensemble"]
            return EnsembleTask.req(
                self,
                config_file=self.config_file,
                workflow=self.workflow,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=ensemble_index,
            )
        else:
            # This is the workflow container (branch=-1) - no direct requirements
            return []
    
    # ========================================================================
    # HTCondor Workflow Methods (only used when workflow="htcondor")
    # ========================================================================
    
    def create_branch_map(self):
        """Define workflow branches - one per ensemble"""
        # Always return branch map since we inherit from HTCondorWorkflow
        num_ensembles: int = self.estimator_config.expands.ensembles.num_ensembles or 1
        num_ensembles = max(1, num_ensembles)
        return {
            ensemble_index: {"ensemble": ensemble_index}
            for ensemble_index in range(num_ensembles)
        }
    
    def workflow_requires(self):
        """Workflow-level requirements"""
        return {}
    
    def htcondor_output_directory(self):
        """Directory for HTCondor job logs"""
        return law.LocalDirectoryTarget(
            f".law/htcondor/{self.task_family}/{self.estimator}_{self.systematic}"
        )
    
    def htcondor_bootstrap_file(self):
        """Bootstrap script for remote environment setup"""
        return law.util.rel_path(__file__, "../../setup.sh")
    
    def htcondor_job_config(self, config, job_num, branches):
        """Configure HTCondor resources for EnsembleTasks"""
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
        return {"outputs": law.LocalDirectoryTarget(f"{self.abs_results_path}/systematic_results.json")}

    def run(self):
        systematic_results = SystematicResults()

        for ensemble_output in self.input():
            ensemble_result = EnsembleResults.from_json(ensemble_output["outputs"].path)
            systematic_results.ensembles.append(ensemble_result)

        systematic_results.to_json(self.output()["outputs"].path)
