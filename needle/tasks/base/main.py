from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Type
from urllib.parse import urlencode

import luigi
from omegaconf import OmegaConf, DictConfig

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_utils import compare_configs, initialize_hydra_config
from needle.utils.logging import ColorFormatter, LogOnce
from needle.utils.dataclass import SerializableDataclass, dataclass
from dataclasses import field

logger = ColorFormatter.get_logger("dag")


@dataclass
class NoteMetadata(SerializableDataclass):
    checkpoint_path: str
    task_type: str
    fold_index: int
    ensemble_index: int
    estimator_name: str
    systematic_name: str


@dataclass
class SnapshotTree(SerializableDataclass):
    config_snapshot: DictConfig
    nodes: dict[str, NoteMetadata] = field(default_factory=dict)


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

    def run(self) -> None:
        self.print_config_path_once()
        logger.info("Processing...")
        nodes = {}

        for estimator_task in self.requires():
            estimator_name = estimator_task.estimator
            logger.info(f"|  Estimator:    {estimator_name}")

            for systematic_task in estimator_task.requires():
                systematic_name = systematic_task.systematic
                logger.info(f"|    Systematic: {systematic_name}")

                for ensemble_task in systematic_task.requires():
                    ensemble_idx = ensemble_task.ensemble
                    logger.info(f"|      Ensemble: {ensemble_idx}")

                    for fold_idx, fold_task in enumerate(ensemble_task.requires()):
                        (training_task,) = fold_task.requires()
                        training_output = training_task.output_as_dict(training_task.output())

                        checkpoint_path = self._find_checkpoint(training_output)

                        node_id = urlencode(
                            {
                                "est": estimator_name,
                                "syst": systematic_name,
                                "ensem": ensemble_idx,
                                "fold": fold_idx,
                            }
                        )

                        nodes[node_id] = NoteMetadata(
                            checkpoint_path=checkpoint_path,
                            task_type="fold",
                            fold_index=fold_idx,
                            ensemble_index=ensemble_idx,
                            estimator_name=estimator_name,
                            systematic_name=systematic_name,
                        )

        snapshot = SnapshotTree(
            nodes=nodes,
            config_snapshot=OmegaConf.to_container(self.config, resolve=True),  # type: ignore
        )
        snapshot.to_json(self.output()["dag_snapshot"].path)
        logger.info(f"DAG snapshot saved to {self.output()['dag_snapshot'].path}")

    def _find_checkpoint(self, training_output: Dict[str, Any]) -> str:
        ckpt_path = training_output["ckpt"].path

        if Path(ckpt_path).exists():
            return ckpt_path

        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
