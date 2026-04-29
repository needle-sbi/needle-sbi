"""
Main entry point for NEEDLE training pipeline. 
The `MainTask` herein is a `WrapperTask` specifically designed
with no `def output` method implemented

The workflow orchestration here is designed to follow the following 
hierarchy:

  MainTask (entry point)
    └─ SnapshotTask (ensures snapshot creation)
         └─ EstimatorTask(s) (training pipeline)
              └─ SystematicTask(s)
                   └─ EnsembleTask(s)
                        └─ FoldTask(s)
"""
import law
import luigi

from law_tasks.mixins import HydraMixin
from law_tasks.snapshot import SnapshotTask

class MainTask(law.WrapperTask, HydraMixin):
    """
    Main orchestration task for NEEDLE pipeline.

    Usage:
        law run MainTask --config-file conf/config.yaml [--use-htcondor True]
    """
    
    use_htcondor: bool = law.Parameter(
        description="Whether to use HTCondor for EnsembleTasks.",
        default=False,
        significant=False,
    )  # type: ignore
    
    def requires(self):
        """
        Required tasks for MainTask.

        Current requirements:
            └─ SnapshotTask (create snapshot of DAG in .json format)
        """
        return [
            SnapshotTask.req(
                self,
                config_file=self.config_file,
                use_htcondor=self.use_htcondor,
            )]
