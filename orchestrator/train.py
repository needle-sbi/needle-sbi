import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from ml.utils import MLConfig
from ml.data import ParticleBase
from ml.utils import timing
from ml.utils import log_progress
from ml.models.model.transformer import MockTransformer

logger = logging.getLogger("ml")


class TrainingBase:
    def __init__(
            self,
            dataset: ParticleBase,
            config: MLConfig = MLConfig(),
            tensor_board_log_dir: str | None = None,
    ):
        self.config = config
        self.dataset = dataset
        self.tensor_board_writer = SummaryWriter(log_dir=tensor_board_log_dir) if tensor_board_log_dir else None
        
    def count_parameters(self, model: torch.nn.Module) -> int:
        """
        Count the number of parameters in model
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prepare_model(self) -> None:
        """
        Prepare the model for training
        """
        self.model = MockTransformer(
            num_features=len(self.dataset.feature_names),
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
            shuffle=self.dataset.SHUFFLE_ALLOWED,
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

    @timing
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

        self.prepare_model()

        for epoch in range(self.config.total_epoch):

            train_loss = self._train_single_epoch()
            validation_loss = self._validate_single_epoch()

            train_loss_list.append(train_loss)
            validation_loss_list.append(validation_loss)

            if epoch > self.config.warmup:
                self.scheduler.step(validation_loss)

            best_validation_loss = min(best_validation_loss, validation_loss)

            if self.tensor_board_writer:
                self.tensor_board_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tensor_board_writer.add_scalar("Loss/validation", validation_loss, epoch)

            logger.info(f"Epoch {epoch + 1}/{self.config.total_epoch} complete | Best Loss: {best_validation_loss:.4f}")

        return {
            "best_validation_loss": best_validation_loss,
        }
