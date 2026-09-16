from abc import abstractmethod

import law

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("dag")


class LocalWorkflow(law.LocalWorkflow):
    """Thin wrapper around :class:`law.LocalWorkflow` that exists mainly as boilerplate.

    Subclasses must implement :meth:`create_branch_map` and :meth:`run`.
    Sets ``local_workflow_require_branches = True`` so that the workflow waits
    for all branches to complete before it is considered done.

    Note:
        This class exists in order to implement shared methods for Workflows that are independent of
        htcondor and slurm.
    """

    local_workflow_require_branches: bool = True

    @abstractmethod
    def create_branch_map(self) -> None:
        pass

    def workflow_requires(self):
        reqs = super().workflow_requires()
        return reqs

    @abstractmethod
    def run(self) -> None:
        pass
