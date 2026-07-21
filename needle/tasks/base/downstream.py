from __future__ import annotations

from functools import cached_property
from itertools import product
from typing import Any, Dict, NamedTuple, Type

import luigi
from omegaconf import DictConfig, OmegaConf

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_schema import DownstreamTaskConfig
from needle.utils.config_utils import hydra_instantiate
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("downstream")


class BranchTuple(NamedTuple):
    name: str
    parameters: Dict[str, Any]


class BaseDownstreamMixin(HydraParamsMixin):
    """Backend-agnostic mixin for DownstreamTask.

    Provides shared properties and helpers. Does NOT inherit from ``luigi.Task``
    so backends can compose it freely with their own task base class.
    """

    downstream: str = luigi.Parameter(
        description="Name of the downstream Task to run",
        significant=True,
    )  # type: ignore
    results_path: str = luigi.Parameter(
        description="Directory where results are stored",
        default="runs",
        significant=False,
    )  # type: ignore

    @property
    def downstream_config(self) -> DownstreamTaskConfig:
        if not self.config.downstream_tasks:
            raise ValueError(f"No entries were found for downstream Tasks: {self.config.downstream_tasks}")
        return self.config.downstream_tasks[self.downstream]

    @property
    def downstream_results_path(self) -> str:
        if self.config.results_path_downstream:
            return self.config.results_path_downstream
        if self.results_path:
            return self.results_path
        return self.config.results_path or self.results_path

    def _snapshot_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _snapshot_task_class()")

    @cached_property
    def snapshot_path(self) -> str:
        SnapshotTask = self._snapshot_task_class()
        snapshot = SnapshotTask(
            results_path=self.results_path,
            config_file=self.config_file,
            hydra_overrides=self.hydra_overrides,
        )
        return snapshot.output()["dag_snapshot"].path  # type: ignore

    def create_branch_map(self) -> Dict[int, BranchTuple]:
        expands = self.downstream_config.expands
        if not expands:
            return {0: BranchTuple(name="default", parameters={})}

        keys = list(expands.keys())
        values = list(expands.values())
        branch_map = {}
        for i, combination in enumerate(product(*values)):
            params = dict(zip(keys, combination))
            from urllib.parse import urlencode
            branch_name = urlencode(sorted(params.items()))
            branch_map[i] = BranchTuple(name=branch_name, parameters=params)
        return branch_map

    def downstream_task(self, branch_id: int) -> luigi.Task:
        """Instantiate the wrapped external task from config for the given branch."""
        base_args: DictConfig = OmegaConf.to_container(
            self.downstream_config.args,
            resolve=True,
        )  # type: ignore
        branch_map = self.create_branch_map()
        branch_args: Dict[str, Any] = branch_map[branch_id].parameters

        merged_args = DictConfig({**base_args, **branch_args})

        return hydra_instantiate(merged_args, snapshot_path=self.snapshot_path)
