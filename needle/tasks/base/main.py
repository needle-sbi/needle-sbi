from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Type

import luigi
from omegaconf import OmegaConf
import json

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_utils import compare_configs, initialize_hydra_config
from needle.utils.logging import ColorFormatter, LogOnce
from needle.tasks.base.estimator import BaseEstimatorTask

logger = ColorFormatter.get_logger("dag")


class BaseMainTask(HydraParamsMixin, luigi.Task):
    """Backend-agnostic base for MainTask.

    Caches the resolved Hydra config, fans out to EstimatorTasks, then on
    completion walks the DAG to write ``dag_snapshot.json``.
    """

    results_path: str = luigi.Parameter(
        description="Root directory where results are saved.",
        default="runs",
        significant=False,
    )  # type: ignore
    strict_config: str = luigi.Parameter(
        description="Config conflict strictness: IGNORE, WARN, or RAISE.",
        default="WARN",
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        if self.results_path != "runs":
            if self.config.results_path:
                LogOnce(logger).warn_once(
                    f"Conflicting value for arg `--results-path`. Config indicates '{self.config.results_path}' "
                    f"while CLI arg is '{self.results_path}'. The CLI value takes precedence."
                )
            return Path(os.path.abspath(self.results_path))
        if self.config.results_path:
            return Path(os.path.abspath(self.config.results_path))
        return Path(os.path.abspath(self.results_path))

    def _estimator_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _estimator_task_class()")

    def requires(self) -> List[Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        cache_config_filepath = Path(os.path.join(self.abs_results_path, "config.yaml"))
        self.config._resolved = True

        if cache_config_filepath.exists():
            cached_config = initialize_hydra_config(
                str(cache_config_filepath.parent),
                cache_config_filepath.stem,
            )
            config_diff = compare_configs(self.config, cached_config)

            if config_diff:
                msg = (
                    "The cached version of your config does not match the new instance. Training results "
                    "might differ based on the changed lines. Use `--remove-output` to delete the cached "
                    "files from the previous run if you want a fresh run. Offending entries are (new, old):"
                    f"\n{config_diff}"
                )
                match self.strict_config.upper():
                    case "WARN":
                        logger.warning(msg)
                    case "RAISE":
                        raise RuntimeError(msg)
                    case "IGNORE":
                        pass
                    case _:
                        raise ValueError(
                            f"Unknown value {self.strict_config} for Parameter 'strict_config'. "
                            "Must be one of IGNORE, WARN, RAISE."
                        )

        with open(cache_config_filepath, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.structured(self.config), resolve=True))

        EstimatorTask = self._estimator_task_class()
        return [
            EstimatorTask(
                config_file=cache_config_filepath,
                hydra_overrides=self.hydra_overrides,
                estimator=estimator_key,
                results_path=self.abs_results_path,
            )
            for estimator_key in self.config.estimators.keys()
        ]

    def output(self) -> Dict[str, Any]:  # type: ignore
        return {"dag_snapshot": luigi.LocalTarget(f"{self.abs_results_path}/dag_snapshot.json")}

    def snapshot_as_dict(self) -> Dict[str, str]:
        snapshot_nested: Dict[str, Dict[str, str]] = {}
        snapshot_flattened: Dict[str, str] = {}
        estimator_task: BaseEstimatorTask

        for estimator_task in self.requires():
            snapshot_nested[estimator_task.estimator] = estimator_task.input_model_paths

        for estimator, est_values in snapshot_nested.items():
            for key, value in est_values.items():
                snapshot_flattened[f"est={estimator}&{key}"] = value

        return snapshot_flattened

    def run(self) -> None:
        self.print_config_path_once()

        file = self.output()['dag_snapshot'].path

        with open(file, "w") as f:
            json.dump(self.snapshot_as_dict(), f, indent=4, sort_keys=True, default=str)

        logger.info(f"DAG snapshot saved to {file}")

    def _find_checkpoint(self, training_output: Dict[str, Any]) -> str:
        ckpt_path = training_output["ckpt"].path

        if Path(ckpt_path).exists():
            return ckpt_path

        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
