# Orchestrator for NEEDLE ML workflows

![pipeline](https://gitlab.desy.de/needle/orchestrator/badges/dev/pipeline.svg)
![coverage](https://gitlab.desy.de/needle/orchestrator/badges/dev/coverage.svg)

## Overview

The orchestrator ties together the [ml](https://gitlab.desy.de/needle/ml) and [preprocessor](https://gitlab.desy.de/needle/preprocessor) submodules with a [LAW](https://github.com/riga/law)-based workflow manager and [Hydra](https://hydra.cc/) configuration. It supports k-fold training, experiment tracking, and in the future also remote job submission via a range of job scheduling technologies (HTCondor, Slurm, etc.).


## Getting started

> **Note:** The active development branch is `dev`, not `main` currently.

### 1. Clone

```bash
git clone git@gitlab.desy.de:needle/orchestrator.git --recurse-submodules
cd orchestrator
git checkout fair_universe_demo
git submodule update --init --recursive
```

### 2. Install dependencies

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

Install it with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell or source the uv environment to make it active in your current shell, as stated by
uv when you run the script.

Download and install the python dependencies:

```bash
uv python pin 3.12   # Use recommended python version (higher is also usually okay)
uv sync              # install runtime dependencies
uv sync --group dev  # include dev tools (pytest, black, flake8, etc.)
```

### 3 Activate the environment

```bash
source .venv/bin/activate
source setup.sh    # sets up the LAW environment variables
law index          # indexes available LAW tasks in the law.cfg file
```

### 4. FAIR Universe Dataset

```bash
pytest  # runs non-slow, non-benchmark tests by default
```

Some training tests require the `FAIR_UNIVERSE_DATA` environment variable pointing to a parquet file:

```bash
export FAIR_UNIVERSE_DATA=/path/to/fair_universe/data.parquet
```

If you dont want to use the full FAIR Universe Dataset but only the test dataset (1000 events) located at `examples/fair_universe_demo/test_data/` you still need to set the environment variable:

```bash
export FAIR_UNIVERSE_DATA=""
```

The full dataset can be obtained from codabench

```bash
cd /path/to/desired/directory  # can be in the same repo
wget -O public_data.zip https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/
unzip public_data.zip
export FAIR_UNIVERSE_DATA="</path/to/desired/directory>/input_data/train/data/data.parquet
```

It is recommended to add the `$FAIR_UNIVERSE_DATA` environment variable to your `~/.bashrc` (or equivalent) to have a persistent setup each time you reload your shell.

## Running LAW tasks

The three top-level Tasks you can use are `MainTask` (only training), `SnapshotTask` (gather checkpoints in tree structure) and `DownstreamTask` (your hook for running custom luigi Tasks).

> **Note:** if running on ARM Arch Macbook you need to set `--workers 1` to avoid Luigi spawn/pickling issues with patched worker callbacks.

### 1. Training: MainTask

```bash
law run MainTask \
    --config-file conf/config.yaml \        # optional
    --hydra-overrides ""                    # optional (must be a single string)
```

We try to avoid having arguments in the argparser as that makes it difficult to replicate previous work.
Including all the parameters in the `config.yaml` is the better way to go for reproducibility.

See the [LAW documentation](https://law.readthedocs.io/en/latest/) for more details on the law shell parameters.

See the [hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) for more details
on overriding the config from the CLI.

### 2. Model Gathering: SnapshotTask

This Task requires all the trainings to be complete by calling MainTask first. It will then recursively
go through all upstream Tasks and gather the Lightning checkpoints of all the models that were trained.
In this branch of the repo, the resulting `dag_snapshot.json` is a flat dict of all the models. The CLI
args are the same as MainTask:

```bash
law run SnapshotTask \
    --config-file conf/config.yaml \
    --hydra-overrides ""
```

### 3. Custom Luigi Tasks: DownstreamTask

If you are using `law` for your own downstream analysis, you can directly require our NEEDLE Tasks and
build your DAG that way. This would make your whole setup completely uniform. In this case, take care
of registering our and your Tasks in `law.cfg` and ensuring that you share the same `law.cfg` for the
whole DAG.

If instead you use plain luigi and want to append some new downstream post-training Tasks, the easiest
way is to export your Tasks using the NEEDLE `config.yaml` and let NEEDLE (and law) run the luigi Tasks
for you. Unfortunately, running law Tasks from a luigi base is prone to errors which we therefore want
to discourage. Running the whole workflow (training + your luigi Tasks) would look like this

```bash
law run DownstreamTask \
    --downstream "<name-of-your-luigi-task>"  # As defined in config.yaml
```

More info is given in the corresponding part of the documentation.


## Jupyter notebooks

After installing dev dependencies, register the kernel:

```bash
uv run python -m ipykernel install --user --name needle --display-name "NEEDLE"
```

Then select the **NEEDLE** kernel when opening notebooks.

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

## Project structure

The current structure is as follows:
```
orchestrator/
├ conf/                  # hydra configs (datasets, models, trainers, datamodules)
├ container/             # singularity container definitions
├ examples/              # Examples with finished models, configs and more
│  └ fair_universe_demo  # FAIR Universe Example code, config and test data
├ law_tasks/             # LAW workflow tasks (training, fold, ensemble)
├ ml/                    # [submodule] models, datasets, blocks, lightning modules
├ notebooks/             # development notebooks
├ orchestrator/          # config dataclasses, results, MLflow logging
├ preprocessor/          # [submodule] data ingestion, normalisation, utilities
├ tests/                 # Integration tests
├ pyproject.toml         # deps, dev deps and tools
├ setup.sh               # LAW environment setup
└ law.cfg                # LAW task registry
```

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
