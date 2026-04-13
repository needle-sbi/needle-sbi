import os
from functools import cached_property
from pathlib import Path
from typing import Dict

import law
import luigi
from law_tasks.mixins.hydra import HydraMixin
from law_tasks.snapshot import SnapshotTask
from omegaconf import OmegaConf
from orchestrator.config import DownstreamTaskConfig
from orchestrator.config_utils import hydra_instantiate
from orchestrator.luigi import convert_luigi_to_law_targets


class DownstreamTask(HydraMixin, law.Task):
    """Task which wraps an external Task that should run after the main training was performed."""

    downstream: str = law.Parameter(
        description="Name of the downstream Task to run",
        significant=True,
    )  # type: ignore
    results_path: str = law.Parameter(
        description="Directory where results are stored",
        default="runs",
        significant=False,
    )  # type: ignore

    @property
    def downstream_config(self) -> DownstreamTaskConfig:
        if not self.config.downstream_tasks:
            raise ValueError(f"No entries where found for downstream Tasks: {self.config.downstream_tasks}")

        return self.config.downstream_tasks[self.downstream]

    @property
    def snapshot_path(self) -> str:
        return self.requires()["snapshot"].output()["dag_snapshot"].path  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        if self.config.results_path:
            self.results_path = self.config.results_path

        return os.path.abspath(self.results_path)  # type: ignore

    def requires(self) -> Dict[str, SnapshotTask | law.Task]:
        req: Dict[str, law.Task] = {}

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
        return convert_luigi_to_law_targets(self.downstream_task.output())

    def input(self):
        return convert_luigi_to_law_targets(self.downstream_task.input())

    @cached_property
    def downstream_task(self) -> luigi.Task:
        return hydra_instantiate(
            OmegaConf.structured(self.downstream_config.args),
            snapshot_path=self.snapshot_path,
        )

    def run(self) -> None:
        self.downstream_task.run()
