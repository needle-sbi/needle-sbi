from dataclasses import dataclass


@dataclass
class MockTransformerSchema:
    """Typed schema for ml.lightning.models.mock_transformer.MockTransformerModule."""

    _target_: str = "ml.lightning.models.mock_transformer.MockTransformerModule"
    factor: float = 0.1
    patience: int = 10
    init_lr: float = 1e-3


@dataclass
class SimpleMLPSchema:
    """Typed schema for ml.lightning.models.simple_mlp.SimpleMLPModule."""

    _target_: str = "ml.lightning.models.simple_mlp.SimpleMLPModule"
    hidden_dim: int = 64
    init_lr: float = 1e-3


@dataclass
class PaddedDataModuleSchema:
    """Typed schema for ml.lightning.data.padded_datamodule.PaddedDataModule."""

    _target_: str = "ml.lightning.data.padded_datamodule.PaddedDataModule"
    multiprocessing_type: str = "torch"
    batch_size: int = 1024
    n_workers: int = 0
    shuffle_partitions: bool = True
    shuffle_events: bool = True


@dataclass
class NormFlowSchema:
    """Typed schema for ml.lightning.models.norm_flow.NormFlowModule."""

    _target_: str = "ml.lightning.models.norm_flow.NormFlowModule"
    features_dim: int = 8
    context_dim: int = 0
    flow_type: str = "nsf"
    n_transforms: int = 5
    hidden_dim: int = 128
    init_lr: float = 1e-3


@dataclass
class LightningTrainerSchema:
    """Typed schema for lightning.Trainer."""

    _target_: str = "lightning.Trainer"
    max_epochs: int = 1
    log_every_n_steps: int = 50
