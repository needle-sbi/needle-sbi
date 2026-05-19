"""SnapshotTask - Creates a complete snapshot of the trained ensemble DAG.

This module defines the SnapshotTask which is responsible for:
- Collecting all trained model checkpoints from the complete DAG
- Creating a mapping of (estimator, systematic, ensemble, fold) → checkpoint path
- Generating the dag_snapshot.json file for evaluation and inference
- Enabling post-training analysis without re-running training

Output:
- dag_snapshot.json: Contains the full DAG structure with checkpoint paths
- config.yaml: The resolved configuration used for training

Dependencies:
- Waits for all MainTask dependencies to complete
- Runs after the complete training DAG is finished

Usage:
The snapshot can be loaded by NEEDLE's pseudo-models for evaluation and inference:
    from needle.evaluation.pseudo_model import NEEDLE
    model = NEEDLE(snapshot_path="runs/dag_snapshot.json")
"""

import os
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode

import law
from omegaconf import OmegaConf

from needle.law_tasks.main import MainTask
from needle.law_tasks.mixins import HydraMixin
from needle.utils.logging import ColorFormatter
from needle.utils.results import (
    AggregationEdge,
    AggregationMethod,
    DAGSnapshot,
    EnsembleResults,
    ModelNodeMetadata,
)

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
        """Ensure all training is completed by requiring MainTask.

        Also caches the resolved config to the results directory for reference.

        Returns:
            MainTask: Root task that triggers all training.
        """
        cache_config_file = os.path.join(self.results_path, "config.yaml")
        self.config._resolved = True

        with open(cache_config_file, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.structured(self.config), resolve=True))

        return MainTask(
            config_file=self.config_file,
            hydra_overrides=self.hydra_overrides,
            results_path=self.abs_results_path,
        )

    def output(self):
        """Define output target for the DAG snapshot.

        Returns:
            Dict: Dictionary with 'dag_snapshot' file target containing the serialized DAG.
        """
        return {"dag_snapshot": law.LocalFileTarget(f"{self.abs_results_path}/dag_snapshot.json")}

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

    def run(self):
        """Traverse the DAG hierarchy and construct a snapshot with nodes and aggregation edges.

        Creates a DAGSnapshot containing:
        - Nodes: Metadata for each FoldTask (leaf level), with checkpoint paths and metrics
        - Edges: Aggregation operations connecting folds → ensembles → systematics → estimators → root

        Aggregation methods (best, mean, etc.) are determined by config. The snapshot enables
        evaluation/inference without re-running training.

        The DAG hierarchy traversed is:
            MainTask → EstimatorTask → SystematicTask → EnsembleTask → FoldTask
        """
        self.print_config_path_once()

        nodes: Dict[str, ModelNodeMetadata] = {}
        edges: List[AggregationEdge] = []

        # Get configuration for aggregation methods from Hydra config
        agg_config = self.config.aggregation
        fold_agg_method = agg_config.fold_method
        ensemble_agg_method = agg_config.ensemble_method
        systematic_agg_method = agg_config.systematic_method
        estimator_agg_method = agg_config.estimator_method

        # Track all node IDs at each level for aggregation
        all_estimator_nodes = []

        logger.info("Processing...")
        main_task = self.requires()

        # Traverse EstimatorTasks
        for estimator_task in main_task.requires():
            estimator_name = estimator_task.estimator
            logger.info(f"|  Estimator:    {estimator_name}")

            all_systematic_nodes = []

            # Traverse SystematicTasks
            for systematic_task in estimator_task.requires():
                systematic_name = systematic_task.systematic
                logger.info(f"|    Systematic: {systematic_name}")

                all_ensemble_nodes = []

                # Traverse EnsembleTasks
                for ensemble_task in systematic_task.requires():
                    ensemble_idx = ensemble_task.ensemble
                    logger.info(f"|      Ensemble: {ensemble_idx}")

                    ensemble_output = ensemble_task.output()
                    # Load EnsembleResults using SerializableDataclass method
                    ensemble_results = EnsembleResults.from_json(ensemble_output["outputs"].path)

                    all_fold_nodes = []

                    # Traverse FoldTasks (leaf nodes)
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
                    ensemble_node_id = urlencode(
                        {
                            "est": estimator_name,
                            "syst": systematic_name,
                            "ensem": ensemble_idx,
                        }
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
                systematic_node_id = urlencode(
                    {
                        "est": estimator_name,
                        "syst": systematic_name,
                    }
                )
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

        snapshot.to_json(self.output()["dag_snapshot"].path)  # type: ignore
        logger.info(f"DAG snapshot saved to {self.output()['dag_snapshot'].path}")

    def _find_checkpoint(self, fold_output) -> str:
        """Get the checkpoint file path for a fold, verifying it exists.

        Args:
            fold_output: Fold task output dictionary.

        Returns:
            str: Absolute path to the model checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        # fold_output["ckpt"] is the LocalFileTarget for model.ckpt
        ckpt_path = fold_output["ckpt"].path
        if Path(ckpt_path).exists():
            return ckpt_path

        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
