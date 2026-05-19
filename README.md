# NEEDLE – The Workflow Orchestrator for Neural Simulation Based Inference Methods

![pipeline](https://gitlab.desy.de/needle/orchestrator/badges/dev/pipeline.svg)
![coverage](https://gitlab.desy.de/needle/orchestrator/badges/dev/coverage.svg)

## Overview

NEEDLE is a framework for the management and training on NSBI tools. It implements most functionalities
needed to train a pool of ML models in a typical HPE analysis environment, meaning deployment to batch
systems (htcondor or slurm), config management and efficient dataloading.

## Getting started

### Step 1: Astral uv

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. You can also use plain `pip`
if you prefer by replacing the `uv` commands with the corresponding `pip` versions.

Install `uv` with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell or source the uv environment to make it active in your current shell, as stated by
uv when you run the script.

### Step 2: Install from Github

 - Option A: Download the `needle` package with `uv` (no source code)

    ```bash
    uv pip install --no-config "git+ssh://git@github.com/needle-sbi/needle-sbi.git"
    ```

 - Option B: Cloning the whole repo (for devs, adds the source code)

    ```bash
    git clone git@github.com:needle-sbi/needle-sbi.git
    uv python pin 3.12   # Use recommended python version (higher is also usually okay)
    uv sync              # install runtime dependencies
    ```

### Step 3

3.1 Source your newly built python environment with

```bash
source .venv/bin/activate
```

This will unlock the `needle` cli tool that installs the files you need to make your project
work within NEEDLE.

3.2 Initialize your project with

```bash
needle init
```

This will copy:

 - `law.cfg`: law/luigi config file for batch submissions
 - `index`: law index file that lists all the available tasks. Can be updated using `law index`
 - `setup.sh`: script for activating the NEEDLE environment
 - `conf/`: template directory for your config files according to the hydra schema

3.3 Source the `setup.sh` script

```bash
source setup.sh
```

**Note**: Every time you start a new shell you have to source your virtual environment and the `setup.sh`
script.

### FAIR Universe Demo (Optional)

We provide an example of how to implement a full NSBI pipeline within needle. For this, we use the
FAIR Universe dataset. If you dont want to use the full dataset (a few GB), there is a test dataset
(1000 events) already shipped with at `examples/fair_universe_demo/test_data/`. The full dataset can
be obtained from codabench

```bash
cd /path/to/desired/directory  # can be in the same repo
wget -O public_data.zip https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/
unzip public_data.zip
export FAIR_UNIVERSE_DATA="</path/to/desired/directory>/input_data/train/data/data.parquet
```

It is recommended to add the `$FAIR_UNIVERSE_DATA` environment variable to your `~/.bashrc` (or equivalent) to have a persistent setup each time you reload your shell.

## Running LAW tasks

The three top-level Tasks you can use are
 - `MainTask` (only training)
 - `SnapshotTask` (gather checkpoints in tree structure)
 - `DownstreamTask` (your hook for running custom luigi Tasks).

> **Note:** if running on ARM Arch Macbook you need to set `--workers 1` to avoid Luigi spawn/pickling issues with patched worker callbacks.

### 1. Training: MainTask

```bash
law run MainTask \
    --config-file conf/config.yaml \
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

You can directly run `SnapshotTask` without first calling `MainTask`, as the later is already a dependency of the former.

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

## Singularity (now apptainer) containers

Pre-built container definitions are in `containerization/`:

- `singularity_base.def` — dependencies only (Python 3.12 + all packages)
- `singularity_dev.def` — full image with source code copied in

```bash
singularity build needle-base.sif containerization/singularity_base.def
singularity build needle.sif containerization/singularity_dev.def
singularity run needle.sif pytest ml/tests
```

When using `singularity exec` or `singularity shell`, you still need to `source setup.sh` and `law index` manually.

## Project structure

The current structure is as follows:
```
needle-sbi/
├ containerization/      # singularity container definitions
├ examples/              # Examples with finished models, configs and more
│  └ fair_universe_demo  # FAIR Universe Example code, config and test data
├ needle/                # source code
├ tests/
├ pyproject.toml
├ conf/config.yaml       # HYDRA config (can also be located elsewhere)
├ setup.sh
└ law.cfg                # LAW config (distinct from needle config.yaml)
```

## Disclaimer on the use of Artificial Intelligence

The vast majority of the code in this project was written by the NEEDLE core development team. Files
in which the code was generated using AI coding agents are marked as such in their corresponding header.
AI-generated code bits are sometimes used in individual functions but not explicitly marked. The docs
were mainly produced using AI under human supervision and review.
