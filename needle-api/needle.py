"""
NEEDLE - High-level API for model evaluation.
Standalone module - no LAW dependency required.
"""
from pathlib import Path
from typing import Optional, Union, Tuple
import torch
from omegaconf import OmegaConf
import hydra

from ml.lightning.models.pseudo_model import NEEDLE as PseudoModel
from orchestrator.config import MainConfig
from orchestrator.config_utils import initialize_hydra_config
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("needle")


class Config:
    """Configuration manager - reuses orchestrator's Hydra infrastructure"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path).resolve()
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Use the same config loader as LAW tasks (but without LAW dependency)
        self.config: MainConfig = initialize_hydra_config(
            config_dir=str(self.config_path.parent),
            config_name=self.config_path.stem,
        )
        
        logger.info(f"Loaded and resolved config from {config_path}")
        logger.info(f"Config resolved: {self.config._resolved}")
    
    def get(self, key: str, default=None):
        """Get config value by dot-notation key"""
        return OmegaConf.select(OmegaConf.structured(self.config), key, default=default)
    
    def __getattr__(self, name):
        """Allow attribute access to config"""
        if name in ['config', 'config_path']:
            return object.__getattribute__(self, name)
        return getattr(self.config, name)


class Dataset:
    """Dataset loader - uses resolved config from initialize_hydra_config"""
    
    def __init__(
        self, 
        cfg: Config, 
        split: str = "test", 
        fold_index: int = 0,
        estimator: Optional[str] = None
    ):
        self.cfg = cfg
        self.split = split
        self.fold_index = fold_index
        
        # If no estimator specified, use the first one
        if estimator is None:
            estimator = list(cfg.config.estimators.keys())[0]
            logger.info(f"No estimator specified, using: {estimator}")
        
        self.estimator = estimator
        estimator_config = cfg.config.estimators[estimator]
        
        # After initialize_hydra_config, the *_override fields are populated!
        # Check if config was resolved
        if not cfg.config._resolved:
            logger.warning(
                "Config not resolved! The *_override fields may be empty. "
                "This should not happen if using initialize_hydra_config."
            )
        
        # Use the resolved override fields (populated by initialize_hydra_config)
        datamodule_config = estimator_config.datamodule_override
        dataset_config = estimator_config.dataset_override
        
        if datamodule_config is None:
            raise ValueError(
                f"Datamodule config is None for estimator {estimator}. "
                f"Config may not have been properly resolved."
            )
        
        if dataset_config is None:
            raise ValueError(
                f"Dataset config is None for estimator {estimator}. "
                f"Config may not have been properly resolved."
            )
        
        # Get n_folds from estimator config
        n_folds = estimator_config.expands.folds
        
        logger.info(f"Using datamodule: {estimator_config.datamodule}")
        logger.info(f"Using dataset: {estimator_config.dataset}")
        
        # Instantiate datamodule using resolved configs
        self.datamodule = hydra.utils.instantiate(
            datamodule_config,
            dataset_config=dataset_config,
            fold_index=fold_index,
            n_folds=n_folds,
        )
        self.datamodule.setup(stage=split)
        
        logger.info(f"Loaded {split} dataset (fold {fold_index}, estimator {estimator})")
    
    def get_dataloader(self):
        """Get PyTorch DataLoader"""
        if self.split == "train":
            return self.datamodule.train_dataloader()
        elif self.split == "val":
            return self.datamodule.val_dataloader()
        else:
            # Need to implement a .test_dataloader() in
            # ml/lightning/data/padded_datamodule.py 
            # but this is likely a problem as we do not want
            # to reserve and save externally the test dataset.
            return self.datamodule.val_dataloader()
    
    def get_tensor(self, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full dataset as tensors (X, y)"""
        dataloader = self.get_dataloader()
        
        all_x = []
        all_y = []
        
        for i, (x, y) in enumerate(dataloader):
            all_x.append(x)
            all_y.append(y)
            
            if max_samples and (i + 1) * x.shape[0] >= max_samples:
                break
        
        X = torch.cat(all_x, dim=0)
        Y = torch.cat(all_y, dim=0)
        
        if max_samples:
            X = X[:max_samples]
            Y = Y[:max_samples]
        
        logger.info(f"Loaded {X.shape[0]} samples from {self.split} split")
        return X, Y


class Model:
    """NEEDLE ensemble model wrapper"""
    
    def __init__(
        self, 
        snapshot_path: Union[str, Path],
        device: Optional[str] = None
    ):
        self.snapshot_path = Path(snapshot_path)
        
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
        
        self.model = PseudoModel(
            snapshot_path=str(snapshot_path),
            device=device
        )
        logger.info(f"Loaded NEEDLE model from {snapshot_path}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Simple prediction without uncertainty"""
        return self.model.eval(x)
    
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction with uncertainty estimates"""
        return self.model.eval_with_uncertainty(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow model(x) syntax"""
        return self.predict(x)


# Convenience functions
def config(config_path: Union[str, Path]) -> Config:
    """Load configuration using orchestrator's Hydra infrastructure"""
    return Config(config_path)


def dataset(
    cfg: Config, 
    split: str = "test", 
    fold_index: int = 0,
    estimator: Optional[str] = None
) -> Dataset:
    """Load dataset using resolved config"""
    return Dataset(cfg, split, fold_index, estimator)


def model(
    snapshot_path: Union[str, Path],
    device: Optional[str] = None
) -> Model:
    """Load NEEDLE model from snapshot"""
    return Model(snapshot_path, device)


__all__ = [
    'Config',
    'Dataset', 
    'Model',
    'config',
    'dataset',
    'model',
]