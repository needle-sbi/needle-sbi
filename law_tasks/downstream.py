import os
from pathlib import Path
from typing import Dict

import law
import luigi
from omegaconf import OmegaConf

from law_tasks.mixins.hydra import HydraMixin
from law_tasks.snapshot import SnapshotTask
from orchestrator.config import DownstreamTaskConfig
from orchestrator.config_utils import hydra_instantiate


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

    def output(self) -> law.LocalFileTarget:
        return law.LocalFileTarget(os.path.join(self.abs_results_path, f"{self.downstream}.done"))

    def run(self) -> None:
        downstream_task: luigi.Task = hydra_instantiate(
            OmegaConf.structured(self.downstream_config.args),
            snapshot_path=self.snapshot_path,
        )
        if downstream_task.requires() != []:
            raise NotImplementedError(
                "Using the luigi `requires` method in downstream Tasks (tasks that are run after the"
                "ML training) is currently not supported. Instead, encode the dependency between your"
                "downstream Tasks using the `requires` keyword in the config.yaml."
            )
        downstream_task.run()
        self.output().touch()
