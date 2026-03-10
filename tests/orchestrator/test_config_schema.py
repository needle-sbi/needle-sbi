import pytest
from omegaconf import OmegaConf

from orchestrator.registry import (
    validate_datamodule_config,
    validate_model_config,
    validate_trainer_config,
)


def test_validate_model_config_rejects_unknown_field_for_known_target():
    config = OmegaConf.create(
        {
            "_target_": "ml.lightning.models.mock_transformer.MockTransformerModule",
            "factor": 0.1,
            "patience": 10,
            "init_lr": 1e-3,
            "unknown": 123,
        }
    )
    with pytest.raises(ValueError, match="Unknown config field"):
        validate_model_config(config)


def test_validate_datamodule_config_rejects_unknown_field_for_known_target():
    config = OmegaConf.create(
        {
            "_target_": "ml.lightning.data.padded_datamodule.PaddedDataModule",
            "batch_size": 128,
            "unknown": "x",
        }
    )
    with pytest.raises(ValueError, match="Unknown config field"):
        validate_datamodule_config(config)


def test_validate_datamodule_config_rejects_invalid_multiprocessing_type():
    config = OmegaConf.create(
        {
            "_target_": "ml.lightning.data.padded_datamodule.PaddedDataModule",
            "batch_size": 128,
            "multiprocessing_type": "invalid",
        }
    )
    with pytest.raises(ValueError, match="Invalid multiprocessing_type"):
        validate_datamodule_config(config)


def test_validate_trainer_config_rejects_unknown_field_for_known_target():
    config = OmegaConf.create(
        {
            "_target_": "lightning.Trainer",
            "max_epochs": 3,
            "unknown": True,
        }
    )
    with pytest.raises(ValueError, match="Unknown config field"):
        validate_trainer_config(config)


def test_validate_model_config_allows_unknown_target_passthrough():
    config = OmegaConf.create(
        {
            "_target_": "custom.module.Model",
            "foo": "bar",
        }
    )
    validated = validate_model_config(config)
    assert dict(validated) == {"_target_": "custom.module.Model", "foo": "bar"}


def test_validate_simple_mlp_config_accepts_known_fields():
    config = OmegaConf.create(
        {
            "_target_": "ml.lightning.models.simple_mlp.SimpleMLPModule",
            "hidden_dim": 16,
            "init_lr": 1e-3,
        }
    )
    validated = validate_model_config(config)
    assert dict(validated) == {
        "_target_": "ml.lightning.models.simple_mlp.SimpleMLPModule",
        "hidden_dim": 16,
        "init_lr": 1e-3,
    }
