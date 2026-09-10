# Writing the Configuration

This page is a hands-on guide for writing and extending NEEDLE config files. For how Hydra and
Lightning are wired together internally, see
[Lightning and Hydra](lightning_and_hydra_integration.md).

## Why Hydra?

A training pipeline has many moving parts: dataset paths, model hyperparameters, training
duration, cross-validation splits. Hardcoding these makes experiments hard to reproduce and putting
them all in one flat file becomes unwieldy.

Hydra solves this by:

1. Composing configs from multiple YAML files. Mix and match models, datasets, and trainers
   independently.
2. Resolving `_target_` strings to Python classes. The config directly connect your code with
    `needle-sbi` using `hydra.utils.instantiate`.
3. Supports runtime overrides. Swap any config value from the command line without editing
   files. The final config will have the correctly merged settings, keeping a single source of
   truth.

## Config directory layout

```
conf/
├── config.yaml            # main config
├── models/
│   ├── my_model.yaml
│   └── other_model.yaml
├── datamodules/
│   └── my_datamodule.yaml
└── trainers/
    └── default.yaml
```

The main `config.yaml` references the group files by filename stem (without `.yaml`). NEEDLE
resolves these references at startup (See [Step 1 resolution](lightning_and_hydra_integration.md#step-1-references-to-other-config-files)).

## The main `config.yaml` file

These are the allowed top level fields.

| Field | Python Type | Description |
|---|---|---|
| `estimators` | `dict[str, EstimatorConfig]` | Dictionary of models to train |
| `downstream_tasks` | `Optional[dict[str, DownstreamTaskConfig]]` | Dictionary of the DownstreamTasks to run after training. |
| `results_path` | `Optional[str]` | Root output directory for training artifacts. |
| `results_path_downstream` | `Optional[str]` | Root output directory for downstream task outputs. Can use OmegaConf interpolations. |
| `custom_settings` | `Optional[Any]` | Extra settings that you want to access throughout the config files via `${custom_settings.*}`. |

### Minimal `config.yaml`

```yaml
results_path: runs/my_experiment
results_path_downstream: "${results_path}/analysis"

estimators:
  my_estimator:                 # key is freely choosable
    model: my_model
    datamodule: my_datamodule
    trainer: default
```

The values inserted here are validated against `MainConfig`
([`needle/utils/config_schema.py`](../../needle/utils/config_schema.py)).

## Estimator blocks

Instead of calling each training a model, we use the term Estimator to differentiate between a single
neural networks (training by TrainingTask) and the neural surrogate that is potentially the combination
of several folds, ensembles and merging of systematic uncertainties. If you have none of the expansions
for a given estimator, then it reduces back to being a model.

Each **estimator** has these fields. Adding any extra ones is forbidden.

| Field                     | Python Type                 | Description                 |
|---------------------------|-----------------------------|-----------------------------|
| `datamodule`              | `str`                       | Name of the sub-config file for the datamodule field. |
| `datamodule_override`     | `Optional[Any]`             | Dictionary with overrides (must match structure from the datamodule sub-config) |
| `dataset`                 | `Optional[str]`             | Name of the sub-config file for the dataset field. Only if using builtin NEEDLE `LightningDataModules` |
| `dataset_override`        | `Optional[Any]`             | Dictionary with overrides (must match structure from the dataset sub-config) |
| `model`                   | `str`                       | Name of the sub-config file for the model field |
| `model_override`          | `Optional[Any]`             | Dictionary with the overrides (must match structure from the model sub-config) |
| `trainer`                 | `str`                       | Name of the sub-config file for the trainer field |
| `trainer_override`        | `Optional[Any]`             | Dictionary with the overrides (must match structure from the trainer sub-config) |
| `requires`                | `Optional[List[str]]` See [requires block](./hydra_config.md#the-requires-block)       | List of the keys of other estimators. This will require their training to complete before starting this estimator |
| `expands`                 | See [expands block](./hydra_config.md#the-expands-block) | How to multiply this estimator for Systematics, Ensembles and Folds. |

The `*_override` mechanism is explained in more detail in [Building the Config](./lightning_and_hydra_integration.md#building-the-config). In essence, you can override the values of the fields from your sub-configs.

### The `expands` block

Controls how many training tasks are spawned per estimator.

| Field                     | Python Type                 | Description                 |
|---------------------------|-----------------------------|-----------------------------|
| `ensembles`               | See `EnsembleConfig`        | How many ensembles to use. Contains one nested field: `ensembles.num_ensembles` which is an `int`. This is because we want to accommodate different ways of ensembling in the future |
| `systematics`             | See `SystematicConfig`      | How to set up Systematics. Is a dictionary with the same fields as `EstimatorConfig`   |
| `folds`                   | `int`                       | Number of folds             |

Continuing with the example from above, with extra `expands` entries as needed:
```yaml
estimators:
  my_estimator:
    model: my_model
    datamodule: my_datamodule
    trainer: default
    expands:
      folds: 5                  # 5 cross-validation folds
      ensembles:
        num_ensembles: 3        # 3 ensemble members per fold
      systematics:
        nominal: {}             # one systematic variation "nominal" (default)
        high_lr:                # another variation named "high_lr"
          model_override:       # same fields as an estimator, allows you to finely override fields
            lr: 1e-2
```

This config spawns 5 × 3 × 2 = 30 `TrainingTask` instances. Each systematic can override any
component (model, datamodule, dataset, trainer) relative to the base estimator config.

::: {hint}
Each of these 30 combinations gets its own output directory and its own key in
`dag_snapshot.json`, e.g. `est=my_estimator&syst=high_lr&ensem=1&fold=3`. See
[Output directory layout](../setup/usage.md#output-directory-layout) to see how to read this
back into your own code.
:::

::: {hint}
In luigi, parameters are an integral part of a Task's identity. This means that two Task instances
with the same `estimator`/`systematic`/`ensemble`/`fold_index` parameters are considered to be the
same Task by the scheduler. **This applies to the parameter names, not their values**!
:::

### The `requires` block

If one estimator needs to use outputs from another (e.g. a stacked model that takes a trained
first-stage model as input), declare `requires`:

```yaml
estimators:
  first_stage:
    model: base_model
    datamodule: base_datamodule
    trainer: default

  second_stage:
    requires:
      - first_stage
    model: stacked_model
    datamodule: stacked_datamodule
    trainer: default
```

`second_stage` will not begin training until all tasks under `first_stage` are complete. The
checkpoint paths of `first_stage` are made available to `second_stage`'s `FoldTask` via
`self.input_model_paths`.

NEEDLE validates that all `requires` entries name existing estimators and that there are no
circular dependencies at config-load time.

### The `resources` block

Usually, the batch resource requests are handled by `b2luigi` (`settings.json`) or `law` (`law.cfg`)
using their own config files or per-Tas

Both `TrainingTask` and `DownstreamTask` also override `htcondor_settings`/`slurm_settings` as
properties that read the `resources` field from `config.yaml` (per estimator/systematic for
`TrainingTask`, per downstream_task entry for `DownstreamTask`) and merge it over the global
`settings.json`/`configure_b2luigi()` settings, winning on conflicting keys:

```yaml
estimators:
  my_estimator:
    resources:
      request_memory: "4096MB"
      request_cpus: 2
    expands:
      systematics:
        jec_up:
          resources:
            request_memory: "8192MB"  # overrides just this key for this systematic

downstream_tasks:
  my_downstream_task:
    resources:
      request_memory: "2048MB"
```

Keys/values are forwarded verbatim - use whatever `htcondor_settings`/`slurm_settings` keys your
batch system expects. An estimator's `resources` and its active systematic's `resources` are
shallow-merged, with the systematic's keys winning on conflict. `resources` is optional; if unset
or empty, only the global `settings.json`/`configure_b2luigi()` settings apply.

## Estimator groups

In contrast to regular configs blocks as above, groups point to a further sub-config file with the
same name. This allows you to defined self-contained and reusable config entries. 

Each group file is a YAML dict that Hydra merges into the `*_override` field of the estimator.
The only required field is `_target_`, which points to the Python class to instantiate. This is
resolved relative to `$PYTHONPATH`, basically if the class is importable from the root of project.

### `model`

Points to `LightningModule`.

```yaml
_target_: my_package.models.my_model.MyModel
hidden_dim: 256
lr: 1e-3
```

`_target_` must be a fully-qualified Python dotted path importable from the project root. All
other keys are passed as keyword arguments to the class constructor.

### `datamodule`

Points to `lightning.LightningDataModule`.

```yaml
_target_: my_package.data.my_datamodule.MyDataModule
batch_size: 512
num_workers: 4
```

NEEDLE passes `dataset_config`, `fold_index`, and `n_folds` as extra kwargs at runtime — see
[Runtime-injected arguments](#runtime-injected-arguments) below.

### `trainer`

The trainer config instantiates a standard `lightning.Trainer`. Callbacks are listed as a
sequence of instantiable configs:

```yaml
_target_: lightning.Trainer
max_epochs: 100
log_every_n_steps: 10
accelerator: auto
devices: 1
callbacks:
  - _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: "val_loss"
    patience: 20
    mode: "min"
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: "val_loss"
    mode: "min"
    save_top_k: 1
```

Which is equivalent to this python code:

```python
lightning.Trainer(
    max_epochs=100,
    log_every_n_steps=10,
    accelerator="auto",
    devices=1,
    callbacks=[
        lightning.pytorch.callbacks.EarlyStopping(
            monitor: "val_loss",
            patience: 20,
            mode: "min",
        ),
        lightning.pytorch.callbacks.ModelCheckpoint(
            monitor: "val_loss",
            mode: "min",
            save_top_k: 1,
        ),
    ],
)
```

### `dataset`

This is an extra config for using the NEEDLE LightningDatamodules which is validated against `DatasetConfig`:

| Field                   | Python Type                       | Description                           |
|-------------------------|-----------------------------------|---------------------------------------|
| `paths`                 | `str`                             | `glob` pattern matching the files to read |
| `features_columns`      | `Optional[List[str]]`             | List of column names for the features |
| `labels_columns`        | `Optional[List[str]]`             | List of column names for the labels   |
| `format`                | `str`                             | "automatic" or "parquet" or "root"    |
| `dak_reader_kwargs`     | `dict[str, Any]`                  | Extra kwargs for `dask` reader        |
| `max_number_events`     | `int`                             | Either `-1` for all or number of events to read |

The dataset can be specified inline in the estimator config (without a group file):

```yaml
estimators:
  my_estimator:
    dataset_override:
      paths: "/data/my_dataset/*.parquet"
      features_columns: ["pt", "eta", "phi", "mass"]
      labels_columns: ["label"]
      max_number_events: 100000
```

Or referenced by name using a group file (`dataset: my_dataset` → `conf/datasets/my_dataset.yaml`).

(runtime-injected-arguments)=
## Runtime-injected arguments

Your `model`, `datamodule`, and downstream task classes aren't instantiated from the YAML alone.
NEEDLE also passes a handful of extra keyword arguments carrying information that's only known at
runtime.

| Consumer | Extra kwargs passed |
|---|---|
| `model` | `dataset_config`, `input_models` |
| `datamodule` | `dataset_config`, `input_models`, `fold_index`, `n_folds` |
| Downstream task (`args._target_`) | `snapshot_path` |

NEEDLE inspects your class's `__init__` signature and `luigi.Parameter` attributes for downstream tasks,
passes only the kwargs it actually supports and silently drops the rest, logging a warning
(an info message for `snapshot_path`) for whatever gets skipped. Your class only needs to declare
the parameters it actually uses, no `**kwargs` catch-all required.

```python
class MyDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size: int, fold_index: int, n_folds: int):
        ...  # receives fold_index/n_folds automatically; dataset_config and input_models are dropped
```

Current runtime-injected arguments are:

 - `dataset_config`
 
    An additional config group for switching datasets independently of 
    `datamodule`. It adheres to the Config schema for the NEEDLE built-in `LightningDatamodule`
    (See `needle.utils.config_schema.DatasetConfig`).

 - `input_models` 
 
    Provides you with a dictionary view on all the models referenced by the `requires`
    keyword for the corresponding estimator. Meaning if model B depends on A, you can access the path
    to the model A's checkpoint using this dictionary.

    The keys are encoded with `urlencode` (<key>=<value>&...). For example, in the FAIR Universe Demo the input models for the classifier as such:

    ```
    {
      "est=nf_background_1jet&syst=c_0.5&ensem=0&fold=0": "./runs/fair_universe_demo/est__nf_background_1jet/syst__c_0.5/ensem__0/fold__0/model.ckpt",
      "est=nf_background_1jet&syst=c_2.0&ensem=0&fold=0": "./runs/fair_universe_demo/est__nf_background_1jet/syst__c_2.0/ensem__0/fold__0/model.ckpt",
      ...
    }
    ```
      The dict is also stored as `input_model.json` for completeness. For loading checkpoints, refer to the [Checkpoint Loading](https://pytorch-lightning.readthedocs.io/en/1.2.10/common/weights_loading.html#checkpoint-loading) page from the Lightning docs.

 - `snapshot_path` 
 
    Points to the file path for the `dag_snapshot.json` file which lists all trained 
    models and their Lightning checkpoints. See [Accessing trained models](../setup/usage.md#accessing-trained-models).


::: {admonition} Example with LightningModule
:class: tip
```
class MyModel(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        lr: float,
        dataset_config: dict,   # <- optional args injected by hydra
        input_models: dict,     #
    ) -> None:
        feature_columns: List[str] = dataset_config.feature_columns
        upstream_model = MyUpstreamModel.load_from_checkpoint(
            input_models["est=nf_background_1jet&syst=c_2.0&ensem=0&fold=0"]
        )
```
:::

## Downstream task config

Downstream tasks are registered under `downstream_tasks`:

| Field                   | Python Type                       | Description                       |
|-------------------------|-----------------------------------|-----------------------------------|
| `requires`              | `Optional[List[str]]`             | Same mechanism as for estimators  |
| `args`                  | `dict[Any]`                       | Required. Needs at least the `_target_` field as an entry |
| `expands`               | `Optional[dict[str, list[Any]]]`  | How to duplicate this task. Use a descriptive name for each key and use a list of values to iterate over. If passing more than one key-value, then the cartesian product of those keys are used. |

```yaml
downstream_tasks:
  my_analysis:
    requires: ["other_task"]    # wait for other_task before running
    args:
      _target_: my_package.tasks.MyAnalysisTask
      output_path: "${results_path_downstream}/results.json"
    expands:
      variant: ["a", "b", "c"]  # spawns one task per value
```

This will spawn three `DownstreamTask(MyAnalysisTask(variant=...))` instances. See
[Writing Custom Downstream Tasks](downstream_tasks.md) for the full guide.

## Config caching

When `MainTask` first runs, it writes the fully resolved config to
`{results_path}/config.yaml`. All downstream tasks in the same run load from this frozen
snapshot. This ensures reproducibility: even if you modify your YAML files mid-run, running
tasks see the original config.

::: {hint}
To change the config and rerun, either change `results_path` or manually delete the cached
config file.
:::

::: {note}
For a complete working example of a multi-estimator config with systematics and downstream tasks,
see the [FAIR Universe demo](../examples/fair_universe_demo/index.md).
:::
