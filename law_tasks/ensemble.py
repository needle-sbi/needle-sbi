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


class EnsembleTask(law.htcondor.HTCondorWorkflow, law.Task, HydraMixin):
    """
    Ensemble task that can run in two modes:
    - local: Creates and runs FoldTasks in-process (current behavior)
    - htcondor: Submits each FoldTask as a separate HTCondor job (workflow mode)
    
    Inherits from HTCondorWorkflow to enable workflow features, but they are
    only activated when workflow="htcondor".
    """
    rel_results_path = law.Parameter(
        description="Directory where the ensemble training results will be saved.",
        default="runs",
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
        return os.path.abspath(self.rel_results_path) 
    
    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def requires(self):
        """
        Conditional requires based on execution mode.
        - local: Returns FoldTask instances to be run in-process
        - htcondor: Handled by LAW workflow machinery (not used)
        """
        if self.workflow == "local":
            return self._requires_local()
        else:
            # HTCondor workflow mode: LAW handles task creation via create_branch_map
            return []
    
    def _requires_local(self):
        """Local execution mode: explicitly create FoldTask instances"""
        return [
            FoldTask.req(
                self,
                config_file=self.config_file,
                workflow=self.workflow,  # Pass workflow mode through
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=self.ensemble,
                fold_index=fold_index,
            )
            for fold_index in range(self.estimator_config.expands.folds)
        ]
    
    # ========================================================================
    # HTCondor Workflow Methods (only used when workflow="htcondor")
    # ========================================================================
    
    def create_branch_map(self):
        """
        Define workflow branches for HTCondor mode.
        Each branch corresponds to one FoldTask.
        """
        if self.workflow == "local":
            # Don't activate workflow mode - use standard requires() instead
            return None
        
        # HTCondor mode: create one branch per fold
        return {
            fold_index: {"fold_index": fold_index}
            for fold_index in range(self.estimator_config.expands.folds)
        }
    
    def workflow_requires(self):
        """Define workflow-level requirements (used by HTCondorWorkflow)"""
        # No requirements at workflow level
        return {}
    
    def htcondor_output_directory(self):
        """Directory for HTCondor job logs and submission files"""
        return law.LocalDirectoryTarget(
            f".law/htcondor/{self.task_family}/"
            f"{self.estimator}_{self.systematic}_ens{self.ensemble}"
        )
    
    def htcondor_bootstrap_file(self):
        """Bootstrap script to set up environment on remote nodes"""
        # Use the workspace setup.sh
        return law.util.rel_path(__file__, "../../setup.sh")
    
    def htcondor_job_config(self, config, job_num, branches):
        """Configure HTCondor job submission parameters for FoldTasks"""
        config.custom_content.append(("request_cpus", "2"))
        config.custom_content.append(("request_memory", "16000"))  # MB
        config.custom_content.append(("request_runtime", "7200"))  # seconds
        config.custom_content.append(("request_GPUs", "0"))
        config.custom_content.append(("getenv", "True"))
        config.custom_content.append(("universe", "vanilla"))
        
        # Set environment variable for data path
        config.custom_content.append((
            "+Environment",
            '"FAIR_UNIVERSE_DATA=/data/dust/group/atlas/needle/FAIRUnv/'
            'UncertaintyChallenge_2024/ProcessedData_v1_2025-10-03/CombData-part0.parquet"'
        ))
        
        return config

    #def output(self) -> Dict[str, Any]:
    #    os.makedirs(self.abs_results_path, exist_ok=True)
    #    return {"outputs": law.LocalFileTarget(f"{self.abs_results_path}/ensemble_results.json")}
    
    def output(self) -> Dict[str, Any]:
        # Create hierarchical directory structure
        ensemble_dir = os.path.join(
            str(self.abs_results_path),
            self.estimator,
            self.systematic,
            f"ensemble_{self.ensemble}",
        )
        os.makedirs(ensemble_dir, exist_ok=True)
        return {
            "outputs": law.LocalFileTarget(str(ensemble_dir+"/ensemble_results.json"))
        }
    
    def run(self):
        # List of fold outputs that are ensemble inputs by constructions
        fold_outputs = self.input()
        
        # Store the individual fold results
        fold_results = []
        for fold_idx, fold_output in enumerate(fold_outputs):
            # Load FoldResults from each fold's output - i.e. ensemble inputs
            fold_result = FoldResults.from_json(fold_output["outputs"].path)
            fold_results.append(fold_result)

        logger.info(f"Loaded {len(fold_results)} fold results for ensemble {self.ensemble}")
        
        # Create EnsembleResults with the folds list populated
        ensemble_results = EnsembleResults(
            folds=fold_results,  
        )
        # Save the SerializableDataclass method
        ensemble_results.to_json(self.output()["outputs"].path)
        logger.info(f"Saved ensemble results to {self.output()['outputs'].path}")
        #EnsembleResults().to_json(self.output()["outputs"].path)