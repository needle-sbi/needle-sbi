"""
Task for a single ensemble training, includes multiple folds.
"""
import os
from pathlib import Path
from typing import Any, Dict

import law
import luigi

# Load HTCondor contrib for workflow support
law.contrib.load("htcondor")

from law_tasks.fold import FoldTask
from law_tasks.mixins import HydraMixin
from orchestrator.results import EnsembleResults, FoldResults
from orchestrator.config import EstimatorConfig
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(HydraMixin, law.LocalWorkflow, law.htcondor.HTCondorWorkflow):
    """
    Ensemble task that runs multiple FoldTasks.
    
    Defaults to HTCondor workflow (rightmost parent).
    Use --EnsembleTask-workflow local for local testing.
    """
    
    rel_results_path = law.Parameter(
        description="Directory where the ensemble training results will be saved.",
        default="runs",
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
        """Get absolute path to results directory.
        
        Uses __file__ to find workspace root so it works correctly on HTCondor
        worker nodes where cwd is the scratch directory.
        """
        results_path = Path(self.rel_results_path)
        
        if results_path.is_absolute():
            return results_path
        
        # Resolve relative to workspace root using module location
        ensemble_file = Path(__file__)  # law_tasks/ensemble.py
        law_tasks_dir = ensemble_file.parent  # law_tasks/
        workspace_root = law_tasks_dir.parent  # orchestrator/
        
        return (workspace_root / results_path).resolve() 
    
    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def create_branch_map(self):
        """Define workflow branches - one per fold"""
        return {
            fold_index: {"fold_index": fold_index}
            for fold_index in range(self.estimator_config.expands.folds)
        }

    def requires(self):
        """Workflow requires nothing; branches require FoldTask"""
        if self.is_branch():
            # Branch task: require the specific FoldTask
            fold_index = self.branch_data["fold_index"]
            return FoldTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=self.ensemble,
                fold_index=fold_index,
            )
        # Workflow container: no requirements
        return []

    def htcondor_output_directory(self):
        """Directory for HTCondor job logs"""
        return law.LocalDirectoryTarget(
            os.path.join(os.getcwd(), ".law", "htcondor", self.task_family,
                        f"{self.estimator}_{self.systematic}_ens{self.ensemble}")
        )
    
    def htcondor_bootstrap_file(self):
        """Bootstrap script for remote environment setup"""
        # Get absolute path to setup.sh in workspace root
        task_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(task_dir)
        return os.path.join(workspace_root, "setup.sh")
    
    def htcondor_job_config(self, config, job_num, branches):
        """Configure HTCondor resources and logging"""
        config.custom_content.append(("request_cpus", "2"))
        config.custom_content.append(("request_memory", "16000"))
        config.custom_content.append(("request_runtime", "7200"))
        config.custom_content.append(("request_GPUs", "0"))
        config.custom_content.append(("getenv", "True"))
        
        # Note: FAIR_UNIVERSE_DATA is exported in setup.sh
        # Note: Do NOT set transfer_input_files - it breaks LAW's internal file transfer
        # The /data/dust filesystem is shared across worker nodes
        
        # Configure log files for debugging
        log_dir = self.htcondor_output_directory().path
        config.custom_content.append(("should_transfer_files", "YES"))
        config.custom_content.append(("when_to_transfer_output", "ON_EXIT"))
        
        # Set actual log files instead of /dev/null
        branch_name = "_".join(str(b) for b in branches)
        config.log = os.path.join(log_dir, f"job_{job_num}_br{branch_name}.log")
        config.stdout = os.path.join(log_dir, f"job_{job_num}_br{branch_name}.out")
        config.stderr = os.path.join(log_dir, f"job_{job_num}_br{branch_name}.err")
        
        return config

    #def output(self) -> Dict[str, Any]:
    #    os.makedirs(self.abs_results_path, exist_ok=True)
    #    return {"outputs": law.LocalFileTarget(f"{self.abs_results_path}/ensemble_results.json")}
    
    def output(self) -> Dict[str, Any]:
        if self.is_branch():
            # Branch output: individual fold result
            ensemble_dir = os.path.join(
                str(self.abs_results_path),
                self.estimator,
                self.systematic,
                f"ensemble_{self.ensemble}",
            )
            os.makedirs(ensemble_dir, exist_ok=True)
            return {
                "outputs": law.LocalFileTarget(
                    os.path.join(ensemble_dir, f"fold_{self.branch}.json")
                )
            }
        # Workflow output: collection of all branch outputs
        ensemble_dir = os.path.join(
            str(self.abs_results_path),
            self.estimator,
            self.systematic,
            f"ensemble_{self.ensemble}",
        )
        os.makedirs(ensemble_dir, exist_ok=True)
        return law.SiblingFileCollection(
            law.LocalFileTarget(
                os.path.join(ensemble_dir, "fold_{branch}.json")
            )
        )
    
    def run(self):
        # Only branches run - workflow just coordinates
        if not self.is_branch():
            raise Exception("Workflow container should not run")
        
        # Branch task: get result from its single FoldTask
        fold_output = self.input()
        fold_result = FoldResults.from_json(fold_output["outputs"].path)
        
        logger.info(f"Loaded fold {self.branch} result for ensemble {self.ensemble}")
        
        # Create EnsembleResults with single fold
        ensemble_results = EnsembleResults(folds=[fold_result])
        
        # Save the result
        ensemble_results.to_json(self.output()["outputs"].path)
        logger.info(f"Saved ensemble fold {self.branch} results to {self.output()['outputs'].path}")
        #EnsembleResults().to_json(self.output()["outputs"].path)