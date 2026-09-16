from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union


def train_single(
    config_file: Union[str, Path],
    estimator: str,
    results_path: Union[str, Path] = "runs",
    *,
    hydra_overrides: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Train a single estimator directly, bypassing the DAG fan-out.

    Equivalent to ``needle run TrainingTask --backend b2luigi --param single=true
    --param estimator=<estimator> --batch-system "local"``

    Ignores `requires`/systematics/ensembles/folds
    entirely and writes output flat under `results_path` (no `est__/syst__/...`
    nesting), so it can never collide with a real DAG run's outputs in the same
    `results_path`.

    Instantiates and runs a real b2luigi ``TrainingTask`` with ``single=True``,
    reusing its exact training logic (MLflow logging and checkpoint saving). The training
    will be executed locally. For the same single-training option but with batch
    system access you should use ``needle.run()``.

    Note:
        The law backend has no in-process execution, only subprocess `law run`, while
        the b2luigi backend does provide it. Therefore there is no choice in backend
        in this function

    Args:
        config_file: Path to the Hydra config file.
        estimator: Name of the estimator to train (must be a key in `config.estimators`).
        results_path: Root directory for results.
        hydra_overrides: Optional list of Hydra override strings.

    Returns:
        Dict[str, Path]: output paths keyed by ``"ckpt"``, ``"model_config"``, ``"input_models"``.
    """
    from needle.tasks.b2luigi.training import TrainingTask

    task = TrainingTask(
        config_file=str(config_file),  # type: ignore
        hydra_overrides=" ".join(hydra_overrides) if hydra_overrides else "",  # type: ignore
        estimator=estimator,
        results_path=str(results_path),
        single=True,
    )
    task.run()
    outputs = task.output()
    return {name: Path(target.path) for name, target in outputs.items()}
