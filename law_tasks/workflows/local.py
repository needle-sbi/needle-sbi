from abc import abstractmethod

import law

from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class LocalWorkflow(law.LocalWorkflow):
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
