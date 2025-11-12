import os
import platform
from typing import Any

import mlflow
import torch

from orchestrator.config import MainConfig
from orchestrator.results import SerializableDataclass


def log_to_mlflow(
    name: str,
    config: MainConfig,
    law_cli_args: dict[str, str],
    results: SerializableDataclass | dict[str, Any],
    model: torch.nn.Module,
    **kwargs,
):
    if isinstance(results, SerializableDataclass):
        results = results.asdict()

    with mlflow.start_run(
        run_name=name,
        nested=True,
    ):
        mlflow.log_params(config.asdict())
        mlflow.log_params(law_cli_args)
        mlflow.pytorch.log_model(model, name="model")
        mlflow.log_dict(results, artifact_file="results")
        mlflow.log_params(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "hostname": os.uname().nodename,
                "python_version": platform.python_version(),
            }
        )
