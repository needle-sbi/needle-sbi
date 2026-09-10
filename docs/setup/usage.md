# Usage

This page assumes you've completed [Setup](index.md) and have a working `conf/config.yaml`.

::: {admonition} Overview
:class: note

NEEDLE takes your Lightning modules, populates the hyperparameters using Hydra and submits the models
to HPCs.
:::

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
case we named the step `"eval"`. For DownstreamTask, the `downstream_name` is a also a positional 
argument, so passing `needle run DownstreamTask eval ...` will also point to the "eval" entry.
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

This case is safe if you are aware of what will run and what wont. You can also run the full DAG later
on and luigi will pick up the Tasks that ran successfully from this singular TrainingTask.
:::

### CLI args

The NEEDLE CLI has these options

| Flag | Effect |
|---|---|
| `--config-file <path>` | Path to the Hydra config YAML |
| `--results-path <path>` | Root directory for results |
| `--param key=value` | Forward an arbitrary parameter to the task (e.g. `--param downstream=eval`, `--param hydra-overrides="key=value key2=value2"`). Can be repeated. |
| `--backend` | Either `"law"` (default) or `"b2luigi"` |
| `--batch-system` | One of `"local"` (default), `"htcondor"`, `"slurm"` or `"lsd"` |
| `--workers` | An `int` indicating the number of workers to request for this task |

::: {admonition} The `--param` wrapping
:class: info

The `--param` flag takes a single `key=value` pair or just a `value`. You can use `--param` as often
as you want.

```bash
needle run DownstreamTask \
    --param downstream=eval \       # key=value pair
    --param help                    # just value
```

This is equivalent to `law run DownstreamTask --downstream eval --help`. For `law` you can exchange
dashes and underscores, they will all be converted to dashes. For `b2luigi` you must use underscores.

If you are using the `law` backend (default), you can also use the `law` CLI tool directly instead
of `needle run`, which avoids the `--param` wrapping entirely and gets you tab-completion. See the
corresponding [LAW Tasks](../concepts/law_tasks.md) page. There is no equivalent native CLI for
`b2luigi` (yet). The `needle run --backend b2luigi` is currently the only CLI entry point.
:::

More worked examples:

```bash
# Run a single model (named "model_A")
needle run TrainingTask --param estimator=model_A --param single
law run TrainingTask --estimator model_A --single  # same as above
```

```bash
# DownstreamTask, b2luigi backend 
needle run DownstreamTask eval --backend b2luigi
# --> DownstreamTask(downstream="eval")

# passing a hydra override through --param (quote the whole value, spaces stay inside it)
needle run MainTask --param hydra_overrides="estimators.model_A.model_override.lr=0.01"
```

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
                └── input_models.json   # List of models that were used as input
```

For the directories we use the `est__<estimator_name>` and subsequent levels schema. 

::: {note}
You are not expected to walk through this structure by hand. Thats where the `dag_snapshot.json` below comes into
play.
:::

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

The FAIR Universe demo's `HistogramTask.parse_snapshot()` has a good reference implementation.

## Troubleshooting

### Using `--backend law` (default)

 - **`task family '<MissingTask>' not found in index`**

    → Ensure the Task you want to run is indexed in the `index` file. Refresh with `law index`. The modules
    to be indexed must be listed in `law.cfg`. 
    
    Make sure this section exists in your `law.cfg`, as it adds all the needle Tasks to the law index:

    ```cfg
    [modules]
    needle.tasks.law
    ```

 - **`ModuleNotFoundError: No module named 'needle.tasks.law'`**

    → You might have forgotten to run `source setup.sh`. Either this or the modules are broken at import
    and `law` failed to load the Tasks. You can debug this by opening python in the terminal and check if
    you can import the module with the current interpreter.

 - **`Unfulfilled dependencies at RunTime`**

    → LAW expected an output file that doesn't exist. Check which file it reports and look at the
    task that should have created it. Often caused by a crashed run leaving partial outputs.

 - **Task shows as complete but results look wrong**

    → LAW only checks file existence, not correctness. Use `--remove-output 0,a,y` on the relevant
task to force a re-run.

### Using `--backend b2luigi`

 - **`Failed task b2luigi.TrainingTask`**

    In the case where the logs `stdout` shows this setup Error:

    ```
    Setting up the environment
    [0;33mLAW not found. Is your virtual environment active?[0m
    ```

    This means that the worker node was unable to access the proper environment to run the Task.

    → For HTCondor, the default `htcondor_settings` has `{"getenv": "True"}`
    (see `merged_batch_settings()` in `needle.tasks.b2luigi.workflows.common`), 
    which copies the environment used for submission
    (activated venv/conda env, `PATH`, `LAW_HOME`, ...) to the worker node, so this should not
    happen normally. 
    
    It can still occur if:
        - This setting is changed to `{"getenv": "False}"` (e.g by your cluster's HTCondor config)
        - The pool disables or ignores `getenv` for security reasons.

    If you want to stay with  `{"getenv": "False}"`, in order to keep a reproducible environemnt
    or for other reasons, point `env_script` at your own script instead of NEEDLE's `setup.sh`. 
    You can do this in `settings.json` or with `configure_b2luigi(env_script=...)`. For
    `conda`, that script would have to include `conda activate <env>`.

    → For Slurm, `sbatch` already copies the submission environment to the worker by default (no
    `getenv`-equivalent setting needed). If `export=NONE` is set, you might still encounter this error.
