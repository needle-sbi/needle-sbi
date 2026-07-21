from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Type
from urllib.parse import urlencode

import luigi
from omegaconf import OmegaConf

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.logging import ColorFormatter
from needle.utils.results import (
    AggregationEdge,
    AggregationMethod,
    DAGSnapshot,
    EnsembleResults,
    ModelNodeMetadata,
)

logger = ColorFormatter.get_logger("snapshot")


class BaseSnapshotTask(HydraParamsMixin, luigi.Task):
    """Backend-agnostic base for SnapshotTask."""

    results_path: str = luigi.Parameter(
        description="Directory where results are stored",
        default="runs",
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        if self.config.results_path:
            self.results_path = self.config.results_path  # type: ignore
        return Path(os.path.abspath(self.results_path))

    def _main_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _main_task_class()")

    def requires(self) -> Any:
        MainTask = self._main_task_class()
        cache_config_file = os.path.join(self.results_path, "config.yaml")
        self.config._resolved = True

        with open(cache_config_file, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.structured(self.config), resolve=True))

        return MainTask(
            config_file=self.config_file,
            hydra_overrides=self.hydra_overrides,
            results_path=self.abs_results_path,
        )

    def output(self) -> Dict[str, Any]:
        return {
            "dag_snapshot": luigi.LocalTarget(f"{self.abs_results_path}/dag_snapshot.json")
        }

    def run(self) -> None:
        self.print_config_path_once()

        nodes: Dict[str, ModelNodeMetadata] = {}
        edges: List[AggregationEdge] = []

        agg_config = self.config.aggregation
        fold_agg_method = agg_config.fold_method
        ensemble_agg_method = agg_config.ensemble_method
        systematic_agg_method = agg_config.systematic_method
        estimator_agg_method = agg_config.estimator_method

        all_estimator_nodes = []

        logger.info("Processing...")
        main_task = self.requires()

        for estimator_task in main_task.requires():
            estimator_name = estimator_task.estimator
            logger.info(f"|  Estimator:    {estimator_name}")

            all_systematic_nodes = []

            for systematic_task in estimator_task.requires():
                systematic_name = systematic_task.systematic
                logger.info(f"|    Systematic: {systematic_name}")

                all_ensemble_nodes = []

                for ensemble_task in systematic_task.requires():
                    ensemble_idx = ensemble_task.ensemble
                    logger.info(f"|      Ensemble: {ensemble_idx}")

                    ensemble_output = ensemble_task.output()
                    ensemble_results = EnsembleResults.from_json(ensemble_output["outputs"].path)

                    all_fold_nodes = []

                    for fold_idx, fold_task in enumerate(ensemble_task.requires()):
                        node_id = urlencode(
                            {
                                "est": estimator_name,
                                "syst": systematic_name,
                                "ensem": ensemble_idx,
                                "fold": fold_idx,
                            }
                        )

                        fold_output = fold_task.output_as_dict(fold_output=fold_task.output())
                        checkpoint_path = self._find_checkpoint(fold_output)
                        fold_result = ensemble_results.folds[fold_idx]

                        nodes[node_id] = ModelNodeMetadata(
                            checkpoint_path=checkpoint_path,
                            task_type="fold",
                            fold_index=fold_idx,
                            ensemble_index=ensemble_idx,
                            estimator_name=estimator_name,
                            systematic_name=systematic_name,
                            metrics={"val_loss": fold_result.best_validation_loss},
                        )
                        all_fold_nodes.append(node_id)

                    ensemble_node_id = urlencode(
                        {"est": estimator_name, "syst": systematic_name, "ensem": ensemble_idx}
                    )
                    edges.append(
                        AggregationEdge(
                            method=AggregationMethod(fold_agg_method),
                            source_nodes=all_fold_nodes,
                            target_node=ensemble_node_id,
                            metric_key="val_loss" if fold_agg_method == "best" else None,
                        )
                    )
                    all_ensemble_nodes.append(ensemble_node_id)

                systematic_node_id = urlencode({"est": estimator_name, "syst": systematic_name})
                edges.append(
                    AggregationEdge(
                        method=AggregationMethod(ensemble_agg_method),
                        source_nodes=all_ensemble_nodes,
                        target_node=systematic_node_id,
                        metric_key="val_loss" if ensemble_agg_method == "best" else None,
                    )
                )
                all_systematic_nodes.append(systematic_node_id)

            estimator_node_id = urlencode({"est": estimator_name})
            edges.append(
                AggregationEdge(
                    method=AggregationMethod(systematic_agg_method),
                    source_nodes=all_systematic_nodes,
                    target_node=estimator_node_id,
                    metric_key="val_loss" if systematic_agg_method == "best" else None,
                )
            )
            all_estimator_nodes.append(estimator_node_id)

        edges.append(
            AggregationEdge(
                method=AggregationMethod(estimator_agg_method),
                source_nodes=all_estimator_nodes,
                target_node="root",
                metric_key="val_loss" if estimator_agg_method == "best" else None,
            )
        )

        snapshot = DAGSnapshot(
            nodes=nodes,
            edges=edges,
            config_snapshot=OmegaConf.to_container(self.config, resolve=True),
            root_node="root",
        )

        snapshot.to_json(self.output()["dag_snapshot"].path)
        logger.info(f"DAG snapshot saved to {self.output()['dag_snapshot'].path}")

    def _find_checkpoint(self, fold_output: Dict[str, Any]) -> str:
        ckpt_path = fold_output["ckpt"].path
        if Path(ckpt_path).exists():
            return ckpt_path
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
