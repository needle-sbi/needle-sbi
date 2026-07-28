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

## Quick test

Try out an absolutely minimal example with the default config (with a mock transformer model) using

```bash
needle run
```

## Running your first task

The two entry points you will use most are:

### `MainTask`: only training

```bash
needle run MainTask \
    --config-file examples/fair_universe_demo/conf/config.yaml
```

This triggers the full training pipeline: all estimators, their systematic variants, ensemble
members, and cross-validation folds. At the end `MainTask` itself writes `dag_snapshot.json`
which maps each trained model to its checkpoint path.

### `DownstreamTask`: training + post-run analysis

```bash
needle run DownstreamTask \
    --config-file examples/fair_universe_demo/conf/config.yaml \
    --param downstream="eval"
```

The `downstream` parameter names one of the keys in `downstream_tasks` inside your config, in this
case we named the step `"eval"`.
NEEDLE automatically runs the training before running the analysis.
More in the [Downstream Tasks](../concepts/downstream_tasks.md) page.

### CLI args

The NEEDLE CLI has these options

| Flag | Effect |
|---|---|
| `--config-file <path>` | Path to the Hydra config YAML |
| `--results-path <path>` | Root directory for results |
| `--param key=value` | Forward an arbitrary parameter to the task (e.g. `--param downstream=eval`, `--param hydra-overrides="key=value key2=value2"`). Can be repeated. |

::: {admonition} The `--param` wrapping
:class: info

In order to have a shared CLI tool for both backends, we introduce an extra layer for differentiating
pure `needle-sbi` args from the ones passed to either law or b2luigi. If you are using the `law` 
backend (default), you can also use the `law` CLI tool instead, which avoids the `--param` wrapping.
See the corresponding [LAW Tasks](../concepts/law_tasks.md) page.

The `--param` flag takes a single `key=value` pair or just a `value`. You can use `--param` as often
as you want.

```bash
needle run DownstreamTask \
    --param downstream=eval \       # key=value pair
    --param help                    # just value
```

This is equivalent to `law run DownstreamTask --downstream eval --help`. For `law` you can exchange
dashes and underscores, they will all be converted to dashes. For `b2luigi` you must use underscores.
:::

## Output directory layout

After a successful run, outputs land under `results_path` from your config

```
runs
├── config.yaml                         # Resolved config snapshot (frozen at run time)
├── dag_snapshot.json                   # Map of all the checkpoints for easy cataloging
└── est__model_A
    └── syst__nominal
        └── ensem__0
            └── fold__0
                ├── model.ckpt          # Last checkpoint
                ├── model_config.yaml   # Exact config used to train this model
                └── input_models.json   # List of models used as input
```

For the directories we use the `est__<estimator_name>` and subsequent levels schema.

## Accessing trained models

The snapshot JSON has the following structure. Read it in your python scripts to access them.

```json
{
    "est=model_A&syst=nominal&ensem=0&fold=0": "./runs/default/est__model_A/syst__nominal/ensem__0/fold__0/model.ckpt"
}
``` 

The schema uses `=` for `key=value` separation and `&` for level separation. More precisely: `est=<my_estimator>&syst=<my_systematics>...`.
The key is produced using `urllib.parse.urlencode` and can be unfurled using `urllib.parse.parse_qs`.

```pycon
>>> from urllib.parse import parse_qs
>>> parse_qs('est=model_A&syst=nominal&ensem=0&fold=0')
{'est': ['model_A'], 'syst': ['nominal'], 'ensem': ['0'], 'fold': ['0']}
```

The FAIR Universe demo's `HistogramTask.parse_snapshot()` is a good reference implementation

## Troubleshooting

**`ModuleNotFoundError: No module named 'needle.tasks.law'`**
→ You might have forgotten to run `source setup.sh`. Either this or the modules are broken at import
 and `law` failed to load the Tasks.

**`Unfulfilled dependencies at RunTime`**
→ LAW expected an output file that doesn't exist. Check which file it reports and look at the
task that should have created it. Often caused by a crashed run leaving partial outputs.

**Task shows as complete but results look wrong**
→ LAW only checks file existence, not correctness. Use `--remove-output 0,a,y` on the relevant
task to force a re-run.
