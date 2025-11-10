"""
Task for a single fold of the training.
"""

import law
import luigi

from law_tasks.training_base import TrainingBaseTask
from ml.data import PaddedTorchDataset, ParticleDaskChunked
from ml.data.kfold import KFold
from orchestrator import TrainingBase
from orchestrator.mlflow import log_to_mlflow
from orchestrator.results import FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")

type ChunkedDataset = PaddedTorchDataset | ParticleDaskChunked


class FoldTask(TrainingBaseTask):
    fold = luigi.IntParameter(
        description="K-Fold index",
        significant=True,
    )
    multiprocessing_type = law.Parameter(
        description="Which multiprocessing library to use, options are `dask` and `torch`",
        significant=False,
        default="torch",
    )

    def output(self):
        return {
            "logs": law.LocalDirectoryTarget(f"{self.results_dir}/fold_{self.fold}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.results_dir}/fold_{self.fold}/training_output.json"),
        }

    def get_dataset_type(self) -> type[ChunkedDataset]:
        allowed_types = ["dask", "torch"]
        m_type = str(self.multiprocessing_type)

        match m_type:
            case "dask":
                return ParticleDaskChunked
            case "torch":
                return PaddedTorchDataset
            case _:
                raise ValueError(f"Parameter 'multiprocessing_type' must from {allowed_types} but is {m_type}")

    @property
    def training_dataset(self) -> ChunkedDataset:
        kfold = KFold(
            fold_index=self.fold,  # type: ignore
            n_folds=self.config.n_folds,
            is_training=True,
            divisions=self.features_ingestor.array.divisions,
        )
        Dataset = self.get_dataset_type()
        return Dataset(
            features=self.features_ingestor,
            labels=self.labels_ingestor,
            shuffle_partitions=self.config.shuffle_partitions,
            shuffle_events=self.config.shuffle_events,
            random_seed=self.config.random_seed,
            kfold=kfold,
        )

    @property
    def validation_dataset(self) -> ChunkedDataset:
        kfold = KFold(
            fold_index=self.fold,  # type: ignore
            n_folds=self.config.n_folds,
            is_training=False,
            divisions=self.features_ingestor.array.divisions,
        )
        Dataset = self.get_dataset_type()
        return Dataset(
            features=self.features_ingestor,
            labels=self.labels_ingestor,
            shuffle_partitions=self.config.shuffle_partitions,
            shuffle_events=self.config.shuffle_events,
            random_seed=self.config.random_seed,
            kfold=kfold,
        )

    def run(self):
        self.warn_if_device_is_cpu()
        training_base = TrainingBase(
            training_dataset=self.training_dataset,
            validation_dataset=self.validation_dataset,
            config=self.config,
            tensor_board_log_dir=self.output()["logs"].path,
        )
        training_outputs = training_base.train()
        fold_outputs = FoldResults(
            **training_outputs.asdict(),
            fold_index=self.fold,  # type: ignore
            n_folds=self.config.n_folds,
        )
        fold_outputs.to_json(self.output()["outputs"].path)
        log_to_mlflow(
            name=f"fold_{self.fold}",
            config=self.config,
            law_cli_args=dict(self.cli_args()),
            results=fold_outputs,
            model=training_base.model,
        )
