"""
Task for a single fold of the training.
"""

import json
import luigi

from preprocessor.utils import ColorFormatter
from law_tasks.training_base import TrainingBaseTask
from orchestrator import TrainingBase
from ml.data.padded_multiple_chunked import ParticleChunked

logger = ColorFormatter.get_logger("fold")


class FoldTask(TrainingBaseTask):

    fold = luigi.IntParameter(
        description="K-Fold index",
        significant=True,
    )

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
