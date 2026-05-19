from needle.law_tasks.workflows.common import check_batch_system
from needle.law_tasks.workflows.htcondor import HTCondorWorkflow
from needle.law_tasks.workflows.local import LocalWorkflow
from needle.law_tasks.workflows.slurm import SlurmWorkflow

__all__ = [
    "LocalWorkflow",
    "HTCondorWorkflow",
    "SlurmWorkflow",
    "check_batch_system",
]
