# Usage

This page assumes you've completed [Setup](index.md) and have a working `conf/config.yaml`.

## Running your first task

There are three entry points you will use:

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

### `TrainingTask`: A single model directly

Since `TrainingTask` is the leaf task that actually runs the Lightning training loop, it can be run
on its own without going through `MainTask` at all. This is useful when you're iterating on or debugging
one specific model and don't want to wait for (or think about) the rest of the DAG. The `estimator`
parameter is always required (there's no name to infer it from otherwise).

```bash
needle run TrainingTask \
    --param estimator=model_A \
    --param single
```

or, with the `law` backend directly:

```bash
law run TrainingTask --estimator model_A --single
```

::: {admonition} What is kept the same
:class: tip
 - HPC submission (via `law.cfg` or `settings.json`)
 - Config schema
 - Training execution
 - MLFlow logging
:::

::: {admonition} What changes
:class: warning
 - The `requires`, `expands` block of your estimator are completely ignored
 - The estimator's own top-level `model`/`datamodule`/ `dataset`/`trainer` config is used directly.
    You can still use the `*_overrides` at the estimator level, but those from `systematics` will be ignored.
 - Output is written flat, directly under `results_path`, with no `est__/syst__/ensem__/fold__` nesting
:::

::: {admonition} What if I dont use `--single`?
:class: note
The `TrainingTask` will behave like a normal DAG leaf and train the single
`(fold_index, ensemble, systematic)` combination that is passed in.
 - The `requires` block will actually work, and all the upstream estimators will run together with
    their own requirements.
 - The `expands` block will be dropped silently, since the leaf node is not aware of its sibling Tasks.
 - The output will be nested as if the Task ran as part of the whole DAG workflow.

This case is safe if you are aware of what will run and what wont. You can also run the full DAG and
luigi will pick up the Tasks that ran successfully from this singular TrainingTask.
:::

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
