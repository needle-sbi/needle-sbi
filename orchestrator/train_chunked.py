import logging
import torch
from torch.utils.data import DataLoader

from ml.data.padded_multiple_chunked import ParticleDatasetChunked
from ml.util.epoch_timer import timing
from ml.util import log_progress
from orchestrator.train import TrainingBase

logger = logging.getLogger(__name__)


class TrainingBaseChunked(TrainingBase):
    def prepare_dataset(self) -> None:
        self.dataset = ParticleDatasetChunked(
            features=self.features_ingestor.arrays_dict,
            labels=self.labels_ingestor.arrays_dict,
            chunk_size=self.config.chunk_size,
            shuffle_chunks=self.config.shuffle_chunks,
            shuffle_events=self.config.shuffle_events,
            random_seed=self.config.random_seed,
        )

    @timing
    def _train_single_epoch(self) -> float:
        """
        Train the model for a single epoch.
        """
        self.model.train()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        epoch_loss = 0.0

        for i, (x, y) in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            x, y = x.to(self.config.device), y.to(self.config.device)
            output = self.model(x)
            loss = self.loss_function(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()

            epoch_loss += loss.item()

            log_progress(
                step=i + 1,
                total_steps=int(len(self.dataloader) / self.config.batch_size) + 1,
                loss=loss,
            )

        return epoch_loss / len(self.dataloader)
