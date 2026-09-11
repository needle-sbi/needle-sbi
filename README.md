# NEEDLE – The Workflow Orchestrator for Neural Simulation Based Inference Methods

[![Tests](https://github.com/needle-sbi/needle-sbi/actions/workflows/tests.yml/badge.svg)](https://github.com/needle-sbi/needle-sbi/actions/workflows/tests.yml)
[![Docs](https://github.com/needle-sbi/needle-sbi/actions/workflows/docs.yml/badge.svg)](https://needle-sbi.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/needle-sbi.svg)](https://pypi.org/project/needle-sbi/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

NEEDLE organizes the training of large collections of neural networks in a typical HEP analysis
environment, including deployment to batch systems (HTCondor, SLURM and LSF), config management and
dataloading.

Write your models in `PyTorch Lightning`, add a config to manage their dependencies and submit the
training to your HPC!

It supports two interchangeable workflow backends: [LAW](https://law.readthedocs.io/en/latest/) and
[b2luigi](https://b2luigi.belle2.org/index.html).

For everything beyond this quickstart see the full
[NEEDLE Documentation](https://needle-sbi.readthedocs.io/en/latest/).

## Installation

Create or use an existing virtual environment with `python3 -m venv`. Install the `needle` package with

```bash
pip install "git+ssh://git@github.com/needle-sbi/needle-sbi.git"
```

For a guide on how to use `astral-uv` instead of `pip`, please refer to the Installation Guide in the docs.

### Set up the NEEDLE environment

1. Source your newly built python environment to unlock the `needle` cli tool

    ```bash
    source .venv/bin/activate
    ```

2. Initialize your workspace with

    ```bash
    needle init
    ```

    You can also use `needle init --backend law` or `needle init --backend b2luigi` to only
    initialize the workspace with the backend that you want. Otherwise you keep both options open.

3. Source the `setup.sh` script

    ```bash
    source setup.sh
    ```

**Note**: Every time you start a new shell you need to source your virtual environment and the `setup.sh`
script (Steps 1 and 3).

### FAIR Universe Demo (Optional)

This example requires a `git clone` to be included.

We provide an example of how to implement a full NSBI pipeline within needle. For this, we use the
FAIR Universe dataset. If you dont want to use the full dataset (a few GB), there is a test dataset
(1000 events) already shipped with at `examples/fair_universe_demo/test_data/`. The full dataset can
be obtained from codabench via

```bash
cd /path/to/desired/directory  # can be in the same repo
wget -O public_data.zip https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/
unzip public_data.zip
export FAIR_UNIVERSE_DATA="</path/to/desired/directory>/input_data/train/data/data.parquet
```

It is recommended to add the `$FAIR_UNIVERSE_DATA` environment variable to your `~/.bashrc` (or equivalent)
to have a persistent setup each time you reload your shell.

## Running your first Tasks

We refer to the documentation for more information about each parameter. Run the default example (only
training), assuming `conf/config.yaml` is the path to the config:

```bash
b2luigi run MainTask
```

Run post-training analysis Tasks with

```bash
b2luigi run DownstreamTask --param downstream=<my_downstream_task>
```

Once you register everything in the `conf/config.yaml` file.

You can also train a single model directly, without going through the full DAG, by running
`TrainingTask` on its own with `single=true`, which ignores `requires`/`expands` entirely and
writes output flat under `results_path`:

```bash
b2luigi run TrainingTask --param estimator=<my_estimator> --param single=true
```

See the [Usage](https://needle-sbi.readthedocs.io/en/latest/setup/usage.html) docs for details.

## Jupyter notebooks

```bash
uv run python -m ipykernel install --user --name needle --display-name "NEEDLE"
```

Then select the **NEEDLE** kernel when opening notebooks.

## Singularity / Apptainer containers

Container definitions live in `containerization/`:

```bash
singularity build needle-base.sif containerization/singularity_base.def   # deps only
singularity build needle.sif containerization/singularity_dev.def         # + source code
singularity run needle.sif <command>
```

`source setup.sh` and `law index` still need to be run manually inside the container.

## Disclaimer on the use of Artificial Intelligence

The vast majority of the code in this project was written by the NEEDLE core development team. Files
in which the code was generated using AI coding agents are marked as such in their corresponding header.
AI-generated code bits are sometimes used in individual functions but not explicitly marked. The docs
were mainly produced using AI under human supervision and review.
