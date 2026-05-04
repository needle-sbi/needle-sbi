# Setup and Usage

This page covers everything you need to go from a fresh checkout to a running training.

## Prerequisites

You need:
- Python 3.12+
- A virtual environment (`.venv/` at the repo root, created e.g. with `python -m venv .venv`). The
    name of the environment is not relevant, but we will use `.venv` as a convention. You can also use
    any other environment manager like `conda`, as long as it is compatible with `pyproject.toml`.
- (Optional) The FAIR Universe dataset if running that example (see `FAIR_UNIVERSE_DATA` below)

## Installation

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Source the LAW environment script — this registers task modules with LAW
source setup.sh

# 3. (First time only) install the package and dependencies
pip install -e .
```

Or use [astral-uv](https://docs.astral.sh/uv/getting-started/installation/) then run `uv sync`.

**Note:** Why `source setup.sh` and not just `python -m law`? LAW needs to know which Python modules
contain your tasks. `setup.sh` sets the `LAW_HOME` and `LAW_CONFIG_FILE` environment variables
so LAW picks up `law.cfg` automatically. You need to re-run `source setup.sh` every time you
open a new terminal.

## Environment variables

| Variable | Purpose |
|---|---|
| `FAIR_UNIVERSE_DATA` | Path to the directory containing `FAIR_Universe_HiggsML_data.parquet` |
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) to expose to PyTorch (e.g. `0`, `1`, `0,1`) |

Set them in your shell or in a `.env` file:

```bash
export FAIR_UNIVERSE_DATA=/path/to/fair_universe_data
export CUDA_VISIBLE_DEVICES=0
```

## The LAW config file (`law.cfg`)

`law.cfg` lives at the repo root and tells LAW where to find tasks and how to run them. Key sections:

```ini
[core]
modules = law_tasks          # Python package(s) containing your Task classes

[luigi_core]                 # luigi section are formatted as `luigi_<section>`
workers = 4                  # How many tasks can run in parallel locally
local_scheduler = True       # Use the embedded scheduler (no separate daemon needed)

[logging]
luigi-interface = INFO
```

Remote execution (SLURM on KIT Horeka, HTCondor on DESY NAF) is also configured here but is
out of scope for this guide.

## Running your first task

The two entry points you will use most are:

### `SnapshotTask` — train everything and collect checkpoints

```bash
law run SnapshotTask \
    --config-file examples/fair_universe_demo/conf/config.yaml
```

This triggers the full training pipeline: all estimators, their systematic variants, ensemble
members, and cross-validation folds. At the end it writes `dag_snapshot.json` which maps each
trained model to its checkpoint path.

### `DownstreamTask` — run analysis after training

```bash
law run DownstreamTask \
    --downstream eval \
    --config-file examples/fair_universe_demo/conf/config.yaml
```

The `--downstream` flag names one of the keys in `downstream_tasks` inside your config.
LAW automatically ensures all upstream tasks (training, snapshot) are complete before running.

### Common flags

| Flag | Effect |
|---|---|
| `--config-file <path>` | Path to the Hydra config YAML |
| `--hydra-overrides "key=value key2=value2"` | Override config values at runtime |
| `--remove-output 0,a,y` | Delete the output of this specific task so it re-runs. Use carefully. |
| `--workers N` | Override number of parallel workers for this run |
| `--print-status 0` | Print the completion status of the DAG without running anything |

### Quick smoke test with test data

The FAIR Universe demo ships a small test dataset (`test_data/`) so you can verify the pipeline
without the full dataset:

```bash
law run DownstreamTask \
    --downstream eval \
    --config-file examples/fair_universe_demo/conf/config.yaml \
    --hydra-overrides "custom_settings.use_test_data=True"
```

This swaps the data path to `test_data/` and writes outputs to `runs/fair_universe_demo_test/`.

## Output directory layout

After a successful run, outputs land under `results_path` from your config
(default: `runs/fair_universe_demo_fixed_normalization/`):

```
runs/fair_universe_demo_fixed_normalization/
├── config.yaml                    # Resolved config snapshot (frozen at run time)
├── dag_snapshot.json              # Maps every model node → checkpoint path
└── stat_only_histogram_mu_one/    # results_path_downstream
    ├── hist.json                  # Classifier score histograms on JES/TES grid
    ├── neyman.json                # Neyman construction results
    ├── eval.json                  # Evaluation results per pseudo-experiment
    ├── scores/                    # Official FAIR Universe scores
    └── plots/                     # Validation and results plots

# Training checkpoints live inside the results_path tree:
runs/.../est__nf_signal_1jet/syst__c_0p5/ensem__0/fold__0/
    ├── best.ckpt                  # Best checkpoint (monitored by val_loss)
    ├── model_config.yaml          # Exact config used to train this model
    ├── metrics.json               # Final training metrics
    └── logs/                      # TensorBoard logs
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'law_tasks'`**
→ You might have forgotten to run `source setup.sh`. Either this or the modules are broken at import
 and `law` failed to load the Tasks.

**`Unfulfilled dependencies at RunTime`**
→ LAW expected an output file that doesn't exist. Check which file it reports and look at the
task that should have created it. Often caused by a crashed run leaving partial outputs.

**Task shows as complete but results look wrong**
→ LAW only checks file existence, not correctness. Use `--remove-output 0,a,y` on the relevant
task to force a re-run.
