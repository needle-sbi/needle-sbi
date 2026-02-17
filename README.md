# Orchestrator for NEEDLE ML workflows

![pipeline](https://gitlab.desy.de/needle/orchestrator/badges/dev/pipeline.svg)
![coverage](https://gitlab.desy.de/needle/orchestrator/badges/dev/coverage.svg)

## Overview

The orchestrator ties together the [ml](https://gitlab.desy.de/needle/ml) and [preprocessor](https://gitlab.desy.de/needle/preprocessor) submodules with a [LAW](https://github.com/riga/law)-based workflow manager and [Hydra](https://hydra.cc/) configuration. It supports k-fold training, experiment tracking via `MLflow` and `TensorBoard`, and remote job submission via a range of job scheduling technologies (HTCondor, Slurm, etc.).

The current structure is as follows:
```
orchestrator/
|─ conf/                  # hydra configs (datasets, models, trainers, datamodules)
|─ container/             # singularity container definitions
|─ law_tasks/             # LAW workflow tasks (training, fold, ensemble)
|─ ml/                    # [submodule] models, datasets, blocks, lightning modules
|─ notebooks/             # development notebooks
|─ orchestrator/          # config dataclasses, results, MLflow logging
|─ preprocessor/          # [submodule] data ingestion, normalisation, utilities
|─ tests/                 # Integration tests
|- pyproject.toml         # deps, dev deps and tools 
|─ setup.sh               # LAW environment setup
|─ law.cfg                # LAW task registry
```

## Getting started

> **Note:** The active development branch is `dev`, not `main` currently.

### 1. Clone

```bash
git clone git@gitlab.desy.de:needle/orchestrator.git --recurse-submodules
cd orchestrator
git checkout dev
git submodule update --init --recursive
```

### 2. Install dependencies

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync              # install runtime dependencies
uv sync --group dev  # include dev tools (pytest, black, flake8, etc.)
```

### 3 Activate the environment

```bash
source .venv/bin/activate
source setup.sh   # sets LAW environment variables
law index          # indexes available LAW tasks in the law.cfg file
```

### 4. Run tests

```bash
pytest  # runs non-slow, non-benchmark tests by default
```

Some training tests require the `FAIR_UNIVERSE_DATA` environment variable pointing to a parquet file:

```bash
export FAIR_UNIVERSE_DATA=/path/to/fair_universe/data.parquet
```

## Running LAW tasks

```bash
law run TrainingBaseTask                          # single training run
law run EnsembleTask                              # k-fold ensemble
law run FoldTask --fold-index 0                   # single fold
law run TrainingBaseTask --config-file conf/config.yaml  # custom config
```
See the [LAW documentation](https://law.readthedocs.io/en/latest/) for more details.

---

> **Note:** (Only For Levi's local setup and testing)

Quick Fair-Universe example (using `../DATA/fair-universe/data/*.parquet` from this repo):

```bash
law run TrainingBaseTask --config-file conf/config_fair_universe_local.yaml
law run EnsembleTask --config-file conf/config_fair_universe_local.yaml
```

> **Note:** if running on ARM arch Macbook will need to set `--workers 1` to avoid Luigi spawn/pickling issues with patched worker callbacks :

## Jupyter notebooks

After installing dev dependencies, register the kernel:

```bash
uv run python -m ipykernel install --user --name needle --display-name "NEEDLE"
```

Then select the **NEEDLE** kernel when opening notebooks.

## TensorBoard

Training logs are saved under `runs/tensorboard_logs` (which cannot be changed in the config for now).The file paths are managed by LAW. To view them:

```bash
tensorboard --logdir runs/tensorboard_logs
```
or you can open them with VSCode by installing the Tensorboard Extension and then using CMD+SHIFT+P to open the command palette and selecting "Python: Launch Tensorboard". Once prompted, manually set the log directory to `runs/tensorboard_logs`. A new tab with Tensorboard will then open.

## Singularity containers

Pre-built container definitions are in `container/`:

- `singularity_base.def` — dependencies only (Python 3.12 + all packages)
- `singularity_dev.def` — full image with source code copied in

```bash
singularity build needle-base.sif container/singularity_base.def
singularity build needle.sif container/singularity_dev.def
singularity run needle.sif pytest ml/tests
```

When using `singularity exec` or `singularity shell`, you still need to `source setup.sh` and `law index` manually.

## Working with submodules

Push orchestrator and submodule changes together:

```bash
git push --recurse-submodules=on-demand
```

## Documentation

> Note: under development

To build the documentation, follow these steps:

```bash
# install docs dependencies from pyproject [dependency-groups].docs
uv sync --group docs
```

then run the following to build the docs:

```bash
uv run python -m sphinx -T -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

You can view the documentation locally by then running the following:

```bash
open docs/_build/html/index.html
```