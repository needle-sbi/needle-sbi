"""DownstreamTask - Generic wrapper for post-training tasks.

This module defines the DownstreamTask which is responsible for:
- Wrapping external user-defined tasks that run after training completes
- Supporting flexible task dependencies and chaining
- Passing training results to downstream analysis
- Managing multi-branch workflows for complex pipelines

Configuration:
Tasks are configured via the ``downstream_tasks`` key in config.yaml:
    downstream_tasks:
      my_task:
        requires: ["snapshot"]
        args:
          _target_: my.module.MyTask
          output_path: "${results_path_downstream}/output"

Dependency Chain:
- Depends on SnapshotTask and any other declared dependencies
- Can form dependency chains (task A → task B → task C)
- Supports branching with task expansions

Usage:
    law run DownstreamTask --downstream my_task
"""

from __future__ import annotations

from typing import Any, Dict, Type

import law
import luigi

from needle.law_tasks.mixins import CollectOutputMixin, HydraMixin
from needle.tasks.base.downstream import BaseDownstreamMixin, BranchTuple
from needle.utils.logging import ColorFormatter
from needle.utils.luigi_utils import convert_luigi_to_law_targets

logger = ColorFormatter.get_logger("downstream")


class DownstreamTask(BaseDownstreamMixin, CollectOutputMixin, law.LocalWorkflow):
    """Task which wraps an external Task that should run after the main training was performed.

    The task is configured via the ``downstream_tasks`` key in the config.yaml file. Each entry
    under ``downstream_tasks`` is a key that can be passed to the ``--downstream`` CLI argument.
    The corresponding value is a ``DownstreamTaskConfig`` which controls how the task is
    instantiated and what it depends on.
    """

    local_workflow_require_branches: bool = True
    branch_map: Dict[int, BranchTuple]  # type: ignore
    branch: int

    def _snapshot_task_class(self) -> Type[luigi.Task]:
        from needle.law_tasks.snapshot import SnapshotTask  # avoid circular imports

        return SnapshotTask

    def requires(self) -> Dict[str, Any]:
        """Resolve dependencies for this downstream task."""
        SnapshotTask = self._snapshot_task_class()
        req: Dict[str, Any] = {}

        if not self.downstream_config.requires:
            req["snapshot"] = SnapshotTask(
                results_path=self.results_path,
                config_file=self.config_file,
                hydra_overrides=self.hydra_overrides,
            )

        if self.downstream_config.requires:
            for downstream_dep in self.downstream_config.requires:
                req[downstream_dep] = DownstreamTask(
                    results_path=self.downstream_results_path,
                    downstream=downstream_dep,
                    config_file=self.config_file,
                    hydra_overrides=self.hydra_overrides,
                )

        return req

    def output(self) -> Any:
        """Convert the wrapped task's output from Luigi to Law format."""
        if self.is_branch():
            task = self.downstream_task(self.branch)
            return convert_luigi_to_law_targets(luigi_targets=task.output())

        targets = {}
        for branch_id in self.branch_map.keys():
            task = self.downstream_task(branch_id)
            luigi_output = task.output()
            law_output = convert_luigi_to_law_targets(luigi_targets=luigi_output)
            targets[branch_id] = law_output

        return law.TargetCollection(targets)

    def input(self) -> Any:
        """Convert the wrapped task's input from Luigi to Law format."""
        if self.is_branch():
            task = self.downstream_task(self.branch)
            return convert_luigi_to_law_targets(task.input())

        targets = {}
        for branch_id in self.branch_map.keys():
            task = self.downstream_task(branch_id)
            luigi_input = task.input()
            law_input = convert_luigi_to_law_targets(luigi_input)
            targets[branch_id] = law_input

        return law.TargetCollection(targets)

    def run(self) -> None:
        if self.is_workflow():
            return None
        else:
            self.downstream_task(branch_id=self.branch).run()

    def workflow_complete(self) -> bool:  # type: ignore
        for branch_id in self.branch_map.keys():
            task = self.downstream_task(branch_id)
            if not task.complete():
                return False
        return True
