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


class SystematicTask(HydraMixin, law.htcondor.HTCondorWorkflow, law.LocalWorkflow):
    """
    Systematic task that runs multiple EnsembleTasks.    
    By default runs as HTCondor workflow (submits branches to cluster).
    Use --SystematicTask-workflow local to run branches locally.    """
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
        return os.path.abspath(self.rel_results_path)  # type: ignore

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

    def htcondor_output_directory(self):
        """Directory for HTCondor job logs"""
        return law.LocalDirectoryTarget(
            os.path.join(os.getcwd(), ".law", "htcondor", self.task_family,
                        f"{self.estimator}_{self.systematic}")
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
        
        # Set actual log files instead of /dev/null
        log_dir = self.htcondor_output_directory().path
        branch_name = "_".join(str(b) for b in branches)
        config.log = os.path.join(log_dir, f"job_{job_num}_br{branch_name}.log")
        config.stdout = os.path.join(log_dir, f"job_{job_num}_br{branch_name}.out")
        config.stderr = os.path.join(log_dir, f"job_{job_num}_br{branch_name}.err")
        
        return config

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
