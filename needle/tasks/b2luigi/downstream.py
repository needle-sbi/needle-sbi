"""DownstreamTask (b2luigi backend) - Wraps post-training user tasks.

Provides full parity with the LAW ``DownstreamTask`` ``expands`` branching, but
implemented via individual task instances instead of ``law.LocalWorkflow``.

Branch expansion is encoded into the ``branch_params`` Luigi parameter so each
combination gets a distinct task identity (different task ID = separate completion
tracking in the Luigi task registry).
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Type
from urllib.parse import parse_qs, urlencode

import b2luigi
import luigi

from needle.tasks.base.downstream import BaseDownstreamMixin
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("downstream")


class DownstreamTask(BaseDownstreamMixin, b2luigi.Task):
    """b2luigi implementation of DownstreamTask.

    Wraps an external user-defined ``luigi.Task`` that runs after training.

    The task is configured via the ``downstream_tasks`` key in the config.yaml file. Each entry
    under ``downstream_tasks`` is a key that can be passed to the ``--downstream`` CLI argument.
    The corresponding value is a ``DownstreamTaskConfig`` which controls how the task is
    instantiated and what it depends on.

    Uses a URL-encoded parameter for branch expansion
    """

    task_namespace = "b2luigi"

    branch_params: str = luigi.Parameter(
        default="",
        significant=True,
        description=(
            "URL-encoded branch parameters for expansion combos. "
            "Empty string means no expansion (default branch)."
        ),
    )  # type: ignore

    @property
    def branch_parameters(self) -> Dict[str, str]:
        """Decode the ``branch_params`` string into a parameter dict."""
        if not self.branch_params:
            return {}
        return {k: v[0] for k, v in parse_qs(self.branch_params).items()}

    def _main_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.main import MainTask  # avoid circular imports

        return MainTask

    def _upstream_deps(self) -> List[luigi.Task]:
        """Resolve main task or chained downstream dependencies. This is required when using
        plain luigi as Task dependencies must be a flat list.
        """
        MainTask = self._main_task_class()
        deps: List[luigi.Task] = []

        if not self.downstream_config.requires:
            deps.append(
                MainTask(
                    results_path=self.results_path,
                    config_file=self.config_file,
                    hydra_overrides=self.hydra_overrides,
                )
            )
        else:
            for downstream_dep in self.downstream_config.requires:
                deps.append(
                    DownstreamTask(
                        results_path=self.downstream_results_path,
                        downstream=downstream_dep,
                        config_file=self.config_file,
                        hydra_overrides=self.hydra_overrides,
                        branch_params="",
                    )
                )

        return deps

    def requires(self) -> List[luigi.Task]:
        """Build dependency list.

        If this is an expansion root (no ``branch_params`` set but ``expands`` is configured),
        returns the upstream deps and the root ``DownstreamTask`` per expansion combo. This
        single root Task is called the `workflow` in LAW, and the same name is used here.

        If this is a branch instance (``branch_params`` set), just returns upstream deps.
        """
        expands = self.downstream_config.expands

        if expands and not self.branch_params:
            # Root combinator: wait for upstream deps + all branch instances
            keys = list(expands.keys())
            values = list(expands.values())
            branch_tasks: List[luigi.Task] = [
                DownstreamTask(
                    downstream=self.downstream,
                    results_path=self.results_path,
                    config_file=self.config_file,
                    hydra_overrides=self.hydra_overrides,
                    branch_params=urlencode(dict(zip(keys, combo))),
                )
                for combo in product(*values)
            ]  # type: ignore
            return self._upstream_deps() + branch_tasks

        # Branch instance or no expansion: just require upstream deps
        return self._upstream_deps()

    def output(self) -> Any:
        """Point to the output of the luigi Task. The `workflow` Task has no output."""
        if not self.branch_params and self.downstream_config.expands:
            return {}

        task = self.downstream_task(self.branch_parameters)
        return task.output()

    def run(self) -> None:
        """Run the wrapped task for branch instances only, the `workflow` Task is no-op."""
        if not self.branch_params and self.downstream_config.expands:
            return None

        task = self.downstream_task(self.branch_parameters)
        task.run()

    def downstream_task(self, extra_params: Dict[str, Any]) -> luigi.Task:
        """Instantiate the wrapped external task, merging branch parameters into config args."""
        from omegaconf import DictConfig, OmegaConf

        from needle.utils.config_utils import hydra_instantiate

        base_args: DictConfig = OmegaConf.to_container(
            self.downstream_config.args,
            resolve=True,
        )  # type: ignore
        merged_args = DictConfig({**base_args, **extra_params})
        return hydra_instantiate(merged_args, snapshot_path=self.snapshot_path)

    def complete(self) -> bool:
        """For root combinator nodes, complete only when all branch outputs exist."""
        if not self.branch_params and self.downstream_config.expands:
            expands = self.downstream_config.expands
            keys = list(expands.keys())
            values = list(expands.values())
            return all(
                DownstreamTask(
                    downstream=self.downstream,
                    results_path=self.results_path,
                    config_file=self.config_file,
                    hydra_overrides=self.hydra_overrides,
                    branch_params=urlencode(dict(zip(keys, combo))),
                ).complete()
                for combo in product(*values)
            )
        return super().complete()
