from __future__ import annotations

from functools import cached_property
from typing import Type

import luigi

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_schema import DownstreamTaskConfig
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("downstream")


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

    def _main_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _main_task_class()")

    @cached_property
    def snapshot_path(self) -> str:
        MainTask = self._main_task_class()
        main = MainTask(
            results_path=self.results_path,
            config_file=self.config_file,
            hydra_overrides=self.hydra_overrides,
        )
        return main.output()["dag_snapshot"].path  # type: ignore
