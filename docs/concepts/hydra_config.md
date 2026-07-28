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
resolves these references at startup — see [Step 1 resolution](lightning_and_hydra_integration.md#config-resolution-initialize_hydra_config).

## The main `config.yaml` file

These are the **top level fields**. Adding any extra fields is forbidden.

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

## The Estimator field

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
| `ensembles`               | See `EnsembleConfig`        | How many ensembles to use. Contains one nested field: `ensembles.num_ensembles` which is an `int`   |
| `systematics`             | See `SystematicConfig`      | How to set up Systematics. Is a dictionary with the same fields as `EstimatorConfig`   |
| `folds`                   | `int`                       | Number of folds             |


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

## Groups: models, datamodules, trainers

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

NEEDLE passes `dataset_config`, `fold_index`, and `n_folds` as extra kwargs at runtime. If your
datamodule accepts them, it receives them automatically; if not, they are dropped with a warning.

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
| `max_number_of_events`  | `int`                             | Either `-1` for all or number of events to read |

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

::: {info}
For a complete working example of a multi-estimator config with systematics and downstream tasks,
see the [FAIR Universe demo](../examples/fair_universe_demo/index.md).
:::