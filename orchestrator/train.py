import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from preprocessor.ingestion.formatter import Ingestor
from ml.config import MLConfig as Config
from ml.data.padded_multiple import ParticleDataset
from ml.util.epoch_timer import timing
from ml.models.model.transformer import Transformer
from tabulate import tabulate

logger = logging.getLogger(__name__)


class TrainingBase:
    def __init__(
            self,
            features_ingestor: Ingestor,
            labels_ingestor: Ingestor,
            config: Config = Config()
    ):
        self.config = config
        self.features_ingestor = features_ingestor
        self.labels_ingestor = labels_ingestor

    def prepare_dataset(self) -> None:
        self.dataset = ParticleDataset(
            features=self.features_ingestor.dict,
            labels=self.labels_ingestor.dict,
        )

    def count_parameters(self, model: torch.nn.Module) -> int:
        """
        Count the number of parameters in model
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prepare_model(self) -> None:
        """
        Prepare the model for training
        """
        self.model = Transformer(
            d_model=self.config.d_model,
            enc_part_feature_size=self.config.enc_part_feature_size,
            dec_part_feature_size=self.config.dec_part_feature_size,
            max_len=self.config.max_len,
            ffn_hidden=self.config.ffn_hidden,
            n_head=self.config.n_heads,
            n_layers=self.config.n_layers,
            drop_prob=self.config.drop_prob,
            device=self.config.device,
        ).to(self.config.device)

        logger.info(f"Model initialized with {self.count_parameters(self.model):,} trainable parameters")

        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.config.init_lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps
        )

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """
        Property that returns the learning rate scheduler. Can be overridden in subclasses.
        Default: ReduceLROnPlateau
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.config.factor,
            patience=self.config.patience,
        )

    @property
    def loss_function(self) -> torch.nn.CrossEntropyLoss:
        """
        Property that returns the loss function used for training. Can be overridden in subclasses.
        Default: CrossEntropyLoss
        """
        return torch.nn.CrossEntropyLoss()

    @timing
    def _train_single_epoch(self) -> float:
        """
        Train the model for a single epoch.
        """
        self.model.train()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
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

        return epoch_loss / len(self.dataloader)

    def _validate_single_epoch(self) -> float:
        """
        Validate the model for a single epoch.

        NOTE Currently validation occurs on the training dataset.
        """
        self.model.eval()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        epoch_loss = 0.0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.dataloader):
                x, y = x.to(self.config.device), y.to(self.config.device)
                output = self.model(x)
                loss = self.loss_function(output, y)
                epoch_loss += loss.item()

        return epoch_loss / len(self.dataloader)

    def train(self) -> dict:
        """
        Train the model from start to finish
        """
        best_validation_loss = float("inf")

        train_loss_list = []
        validation_loss_list = []

        self.prepare_dataset()
        self.prepare_model()

        for epoch in range(self.config.total_epoch):
            train_loss = self._train_single_epoch()
            validation_loss = self._validate_single_epoch()

            train_loss_list.append(train_loss)
            validation_loss_list.append(validation_loss)

            if epoch > self.config.warmup:
                self.scheduler.step(validation_loss)

            best_validation_loss = min(best_validation_loss, validation_loss)
            table = [
                ["Epoch", f"{epoch + 1}/{self.config.total_epoch}"],
                ["Train Loss", f"{train_loss:.4f}"],
                ["Validation Loss", f"{validation_loss:.4f}"],
                ["Best Validation Loss", f"{best_validation_loss:.4f}"]
            ]
            logger.info("\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

        return {
            "best_validation_loss": best_validation_loss,
        }
