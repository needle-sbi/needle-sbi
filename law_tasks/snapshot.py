"""
Creates a snapshot of the trained ensemble DAG for evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List

import law
from omegaconf import OmegaConf

from law_tasks.main import MainTask
from law_tasks.mixins import HydraMixin
from orchestrator.results import (
    AggregationEdge,
    AggregationMethod,
    DAGSnapshot,
    EnsembleResults,
    ModelNodeMetadata,
)
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("snapshot")


class SnapshotTask(HydraMixin, law.Task):
    """
    Creates a complete snapshot of the trained ensemble DAG.
    This snapshot can be used for evaluation without re-running training.
    """

    results_path: str = law.Parameter(
        description="Directory where results are stored",
        default="runs",
        significant=False,
    )  # type: ignore

    def requires(self):
        """Require MainTask to ensure all training is completed"""
        return MainTask(
            self,
            config_file=self.config_file,
            results_path=self.results_path,
        )

    def output(self):
        return law.LocalFileTarget(f"{self.abs_results_path}/dag_snapshot.json")

    @property
    def abs_results_path(self) -> Path:
        return Path(os.path.abspath(self.results_path))

    def run(self):
        """
        Traverse the entire DAG hierarchy:
        MainTask → EstimatorTask → SystematicTask → EnsembleTask → FoldTask
        """
        nodes: Dict[str, ModelNodeMetadata] = {}
        edges: List[AggregationEdge] = []

        # Get configuration for aggregation methods from Hydra config
        agg_config = self.config.aggregation
        fold_agg_method = agg_config.fold_method
        ensemble_agg_method = agg_config.ensemble_method
        systematic_agg_method = agg_config.systematic_method
        estimator_agg_method = agg_config.estimator_method

        # Get optional weights
        # fold_weights = agg_config.get("fold_weights")
        # ensemble_weights = agg_config.get("ensemble_weights")
        # systematic_weights = agg_config.get("systematic_weights")
        # estimator_weights = agg_config.get("estimator_weights")

        main_task = self.requires()

        # Track all node IDs at each level for aggregation
        all_estimator_nodes = []

        # Traverse EstimatorTasks
        for estimator_task in main_task.requires():
            estimator_name = estimator_task.estimator
            logger.info(f"Processing estimator: {estimator_name}")

            all_systematic_nodes = []

            # Traverse SystematicTasks
            for systematic_task in estimator_task.requires():
                systematic_name = systematic_task.systematic
                logger.info(f"  Processing systematic: {systematic_name}")

                all_ensemble_nodes = []

                # Traverse EnsembleTasks
                for ensemble_task in systematic_task.requires():
                    ensemble_idx = ensemble_task.ensemble
                    logger.info(f"    Processing ensemble: {ensemble_idx}")

                    ensemble_output = ensemble_task.output()
                    # Load EnsembleResults using SerializableDataclass method
                    ensemble_results = EnsembleResults.from_json(ensemble_output["outputs"].path)

                    all_fold_nodes = []

                    # Traverse FoldTasks (leaf nodes)
                    for fold_idx, fold_task in enumerate(ensemble_task.requires()):
                        node_id = (
                            f"estimator_{estimator_name}_"
                            f"systematic_{systematic_name}_"
                            f"ensemble_{ensemble_idx}_"
                            f"fold_{fold_idx}"
                        )

                        fold_output = fold_task.output()

                        # Find checkpoint path
                        checkpoint_path = self._find_checkpoint(fold_output)

                        # Extract metrics from ensemble results
                        fold_result = ensemble_results.folds[fold_idx]

                        nodes[node_id] = ModelNodeMetadata(
                            checkpoint_path=checkpoint_path,
                            task_type="fold",
                            fold_index=fold_idx,
                            ensemble_index=ensemble_idx,
                            estimator_name=estimator_name,
                            systematic_name=systematic_name,
                            metrics={
                                "val_loss": fold_result.best_validation_loss,
                                # "train_loss": fold_result.final_train_loss,
                            },
                        )
                        all_fold_nodes.append(node_id)

                    # Aggregate folds → ensemble
                    ensemble_node_id = (
                        f"estimator_{estimator_name}_" f"systematic_{systematic_name}_" f"ensemble_{ensemble_idx}"
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

                # Aggregate ensembles → systematic
                systematic_node_id = f"estimator_{estimator_name}_" f"systematic_{systematic_name}"
                edges.append(
                    AggregationEdge(
                        method=AggregationMethod(ensemble_agg_method),
                        source_nodes=all_ensemble_nodes,
                        target_node=systematic_node_id,
                        metric_key="val_loss" if ensemble_agg_method == "best" else None,
                    )
                )
                all_systematic_nodes.append(systematic_node_id)

            # Aggregate systematics → estimator
            estimator_node_id = f"estimator_{estimator_name}"
            edges.append(
                AggregationEdge(
                    method=AggregationMethod(systematic_agg_method),
                    source_nodes=all_systematic_nodes,
                    target_node=estimator_node_id,
                    metric_key="val_loss" if systematic_agg_method == "best" else None,
                )
            )
            all_estimator_nodes.append(estimator_node_id)

        # Aggregate estimators → root
        edges.append(
            AggregationEdge(
                method=AggregationMethod(estimator_agg_method),
                source_nodes=all_estimator_nodes,
                target_node="root",
                metric_key="val_loss" if estimator_agg_method == "best" else None,
            )
        )

        # Create snapshot
        snapshot = DAGSnapshot(
            nodes=nodes,
            edges=edges,
            config_snapshot=OmegaConf.to_container(self.config, resolve=True),
            root_node="root",
        )

        self.output().touch()
        snapshot.to_json(self.output().path)
        logger.info(f"DAG snapshot saved to {self.output().path}")

    def _find_checkpoint(self, fold_output) -> str:
        """Find the best or last checkpoint for a fold task"""
        # fold_output["ckpt"] is the LocalFileTarget for model.ckpt
        ckpt_path = fold_output["ckpt"].path
        if Path(ckpt_path).exists():
            return ckpt_path

        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
