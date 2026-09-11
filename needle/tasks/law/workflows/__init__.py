from needle.tasks.law.workflows.common import check_batch_system
from needle.tasks.law.workflows.htcondor import HTCondorWorkflow
from needle.tasks.law.workflows.local import LocalWorkflow
from needle.tasks.law.workflows.slurm import SlurmWorkflow

__all__ = [
    "LocalWorkflow",
    "HTCondorWorkflow",
    "SlurmWorkflow",
    "check_batch_system",
]
