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

from law_tasks.mixins import HydraMixin
from law_tasks.snapshot import SnapshotTask

class MainTask(law.WrapperTask, HydraMixin):
    """
    Main orchestration task for NEEDLE pipeline. 

    Usage:
        law run MainTask --config conf/config.yaml
    """
    def requires(self):
        """
        Required tasks for `MainTasks`. 

        Current requirements:
            └─ SnapshotTask (create snapshot of DAG in .json format)
        """
        return [
            SnapshotTask.req(
                self,
                config_file=self.config_file,
            )]
