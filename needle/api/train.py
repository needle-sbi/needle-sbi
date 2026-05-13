from typing import Any, Dict

import lightning

from needle.utils.config_schema import DatasetConfig
from needle.utils.config_utils import hydra_instantiate


def train_single_lightning_module(
    model_config: Any,
    datamodule_config: Any,
    trainer_config: Any,
    dataset_config: DatasetConfig,
    input_model_paths: Dict[str, str],
) -> lightning.Trainer:
    """Train a single instance of your Lightning module

    Args:
        model_config (Any): Config for your model according to hydra instantiate() protocol
        datamodule_config (Any)
        trainer_config (Any)
        dataset_config (DatasetConfig)
        input_model_paths (Dict[str, str])

    Returns:
        lightning.Trainer: Trained instance of your model
    """
    # 1. Load model
    model = lightning.LightningModule = hydra_instantiate(
        model_config,
        dataset_config=dataset_config,
        input_models=input_model_paths,
    )

    # 2. Load datamodule
    data_module: lightning.LightningDataModule = hydra_instantiate(
        datamodule_config,
        dataset_config=dataset_config,
        input_models=input_model_paths,
    )

    # 3. Load trainer
    trainer: lightning.Trainer = hydra_instantiate(
        trainer_config,
    )
    trainer.fit(model=model, datamodule=data_module)
    return trainer
