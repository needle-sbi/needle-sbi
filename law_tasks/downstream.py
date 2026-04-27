import os
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, NamedTuple
from urllib.parse import urlencode

import law
import luigi
from law_tasks.mixins import CollectOutputMixin, HydraMixin
from law_tasks.snapshot import SnapshotTask
from omegaconf import DictConfig, OmegaConf
from orchestrator.config import DownstreamTaskConfig
from orchestrator.config_utils import hydra_instantiate
from orchestrator.luigi_utils import convert_luigi_to_law_targets
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("downstream")


class BranchTuple(NamedTuple):
    name: str
    parameters: Dict[str, Any]


class DownstreamTask(CollectOutputMixin, HydraMixin, law.LocalWorkflow):
    """Task which wraps an external Task that should run after the main training was performed.

    The task is configured via the ``downstream_tasks`` key in the config.yaml file. Each entry
    under ``downstream_tasks`` is a key that can be passed to the ``--downstream`` CLI argument.
    The corresponding value is a ``DownstreamTaskConfig`` which controls how the task is
    instantiated and what it depends on.

    Config Schema (``DownstreamTaskConfig``):
        args (optional):
            A dictionary of arguments passed to the external task. Must contain ``_target_``
            which is the fully qualified class path of the task to run. All other keys are
            passed as constructor arguments to that task.
        requires (optional):
            A list of other downstream task keys that must complete before this task runs.
            Defaults to None (no dependencies beyond SnapshotTask).

    Dependency Chain:
        If ``requires`` is not set or ``requires=="snapshot"`` (which directly references the SnapshotTask)

            SnapshotTask  # (the root NEEDLE Task)
                └> DownstreamTask(key)  # Wrapper
                    └> YourCustomTask

        If ``requires`` is set, the chain becomes:
            SnapshotTask
                └> DownstreamTask(dep)
                    └> DepTask
                        └> DownstreamTask(key)
                            └> YourCustomTask

    Examples:
        1. **Single standalone task with no dependencies**

            In your `config.yaml`:

            .. code-block:: yaml

                downstream_tasks:
                histogram:
                    args:
                    _target_: my.module.HistogramTask
                    output_path: "${results_path}/hist.json"

            Run from the CLI using:

            .. code-block:: bash
                law run DownstreamTask --downstream histogram

        ----

        2. **A task that requires another downstream task to complete first**

            In your `config.yaml`:

            .. code-block:: yaml

                downstream_tasks:
                histogram:
                    args:
                    _target_: my.module.MyHistogramTask
                    output_path: "${results_path}/hist.json"
                plot:
                    requires: ["histogram"]
                    args:
                    _target_: my.module.MyTask
                    output_path: "${results_path}/plot.pdf"

            Run from the CLI:

            .. code-block:: bash

                law run DownstreamTask --downstream plot
                # histogram will run first automatically

        ----

        3. **Not supported: Referencing a key not defined in downstream_tasks**

            In your `config.yaml`:

            .. code-block:: yaml

                downstream_tasks:
                my_task:
                    requires: ["undefined_task"]  # will raise at runtime
                    args:
                    _target_: my.module.MyTask

        ----

        4. **Not supported: Circular dependencies between downstream tasks**

            In your `config.yaml`:

            .. code-block:: yaml

                downstream_tasks:
                task_a:
                    requires: ["task_b"]  # circular — will deadlock or raise
                    args:
                    _target_: my.module.TaskA
                task_b:
                    requires: ["task_a"]
                    args:
                    _target_: my.module.TaskB

        ----

        5. **Not supported: Omitting the "_target_" field from args**

            In your `config.yaml`:

            .. code-block:: yaml

                downstream_tasks:
                histogram:
                    args:
                    output_path: "${results_path}/hist.json"  # missing _target_, will raise
    """

    downstream: str = law.Parameter(
        description="Name of the downstream Task to run",
        significant=True,
    )  # type: ignore
    results_path: str = law.Parameter(
        description="Directory where results are stored",
        default="runs",
        significant=False,
    )  # type: ignore

    local_workflow_require_branches: bool = True
    branch_map: Dict[int, BranchTuple]  # type: ignore
    branch: int

    @property
    def downstream_config(self) -> DownstreamTaskConfig:
        """Get the configuration for this downstream task.

        Returns:
            DownstreamTaskConfig: Configuration containing task instantiation args and dependencies.

        Raises:
            ValueError: If no downstream tasks are defined in the config.
        """
        if not self.config.downstream_tasks:
            raise ValueError(f"No entries where found for downstream Tasks: {self.config.downstream_tasks}")

        return self.config.downstream_tasks[self.downstream]

    @cached_property
    def snapshot_path(self) -> str:
        """Get the path to the DAG snapshot created by SnapshotTask.

        Cached to avoid recreating the SnapshotTask multiple times.

        Returns:
            str: Path to dag_snapshot.json file.
        """
        snapshot = SnapshotTask(
            results_path=self.results_path,
            config_file=self.config_file,
            hydra_overrides=self.hydra_overrides,
        )
        return snapshot.output()["dag_snapshot"].path  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        """Get the absolute path to the results directory.

        Uses config-specified path if available, otherwise uses the CLI parameter.

        Returns:
            Path: Absolute path to results directory.
        """
        if self.config.results_path:
            self.results_path = self.config.results_path

        return os.path.abspath(self.results_path)  # type: ignore

    def requires(self) -> Dict[str, SnapshotTask | law.Task]:
        """Resolve dependencies for this downstream task.

        If no explicit dependencies are configured, requires SnapshotTask (the root training).
        If dependencies are specified, creates DownstreamTask instances for each dependency.

        Using `workflow_requires` instead of `requires` simplifies the execution logic. All branches
        of this Task will wait for the dependencies as if they were a single Task.

        Returns:
            Dict[str, law.Task]: Named dependencies. Keys are 'snapshot' or dependent task names.
        """
        req: Dict[str, law.Task] = {}

        if not self.downstream_config.requires:
            req["snapshot"] = SnapshotTask(
                results_path=self.results_path,
                config_file=self.config_file,
                hydra_overrides=self.hydra_overrides,
            )

        if self.downstream_config.requires:
            for downstream_dep in self.downstream_config.requires:
                req[downstream_dep] = DownstreamTask(
                    results_path=self.results_path,
                    downstream=downstream_dep,
                    config_file=self.config_file,
                    hydra_overrides=self.hydra_overrides,
                )

        return req

    def output(self):
        """Convert the wrapped task's output from Luigi to Law format.

        Returns:
            Target or nested Target structure converted to Law format.
        """
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

    def input(self):
        """Convert the wrapped task's input from Luigi to Law format.

        Returns:
            Target or nested Target structure converted to Law format.
        """
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

    def create_branch_map(self) -> Dict[int, BranchTuple]:  # type: ignore
        expands = self.downstream_config.expands

        if not expands:
            return {0: BranchTuple(name="default", parameters={})}

        keys = list(expands.keys())
        values = list(expands.values())

        branch_map = {}

        for i, combination in enumerate(product(*values)):
            params = dict(zip(keys, combination))
            branch_name = urlencode(sorted(params.items()))
            branch_map[i] = BranchTuple(name=branch_name, parameters=params)

        return branch_map

    def downstream_task(self, branch_id: int) -> luigi.Task:
        """Instantiate the wrapped external task from config.

        Uses Hydra to instantiate the task class specified in the config's _target_ field,
        passing all other config args and the snapshot_path as constructor arguments.

        Returns:
            luigi.Task: Instantiated external task.
        """
        base_args: DictConfig = OmegaConf.to_container(
            self.downstream_config.args,
            resolve=True,
        )  # type: ignore
        branch_args: Dict[str, Any] = self.branch_map[branch_id].parameters  # type: ignore

        merged_args = DictConfig(
            {
                **base_args,
                **branch_args,
            }
        )

        return hydra_instantiate(
            merged_args,
            snapshot_path=self.snapshot_path,
        )

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
