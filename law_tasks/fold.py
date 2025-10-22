"""
Task for a single fold of the training.
"""

import json

import law
import luigi

from law_tasks.training_base import TrainingBaseTask
from ml.data.padded_multiple_chunked import ParticleChunked
from orchestrator import TrainingBase
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")


class FoldTask(TrainingBaseTask):
    fold = luigi.IntParameter(
        description="K-Fold index",
        significant=True,
    )

    def output(self):
        return {
            "logs": law.LocalDirectoryTarget(f"{self.results_dir}/fold_{self.fold}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.results_dir}/fold_{self.fold}/training_output.json"),
        }

    def run(self):
        training_dataset = ParticleChunked(
            features=self.features_ingestor,
            labels=self.labels_ingestor,
            shuffle_partitions=self.config.shuffle_partitions,
            shuffle_events=self.config.shuffle_events,
            random_seed=self.config.random_seed,
            is_training=True,
            fold_index=self.fold,
            n_folds=self.config.n_folds,
        )
        training_base = TrainingBase(
            dataset=training_dataset,
            config=self.config,
            tensor_board_log_dir=self.output()["logs"].path,
        )
        outputs = training_base.train()

        with open(self.output()["outputs"].path, "w") as f:
            json.dump(outputs, f, indent=4)
