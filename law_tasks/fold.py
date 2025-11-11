"""
Task for a single fold of the training.
"""

import law
import lightning
import luigi

from law_tasks.training_base import TrainingBaseTask
from ml.lightning.mock_transformer import MockTransformerModule
from ml.lightning.padded_datamodule import PaddedDataModule
from orchestrator.results import FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")


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

    def run(self):
        self.warn_if_device_is_cpu()
        model = MockTransformerModule(
            config=self.config,
            tensor_board_log_dir=self.output().get("logs"),
        )
        data_module = PaddedDataModule(
            config=self.config,
            fold_index=self.fold,  # type: ignore
            multiprocessing_type=self.multiprocessing_type,  # type: ignore
        )
        trainer = lightning.Trainer(
            max_epochs=self.config.total_epoch,
        )
        trainer.fit(model=model, datamodule=data_module)
        fold_results = FoldResults(
            **model.results.asdict(),
            fold_index=self.fold,  # type: ignore
            n_folds=self.config.n_folds,
        )
        fold_results.to_json(self.output()["outputs"].path)
