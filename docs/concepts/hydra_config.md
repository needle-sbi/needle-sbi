# Writing the Configuration

This page is a hands-on guide for writing and extending NEEDLE config files. For how Hydra and
Lightning are wired together internally, see
[Lightning and Hydra: Technical Implementation](lightning_and_hydra_integration.md).

---

## Why Hydra?

A training pipeline has many moving parts: dataset paths, model hyperparameters, training
duration, cross-validation splits. Hardcoding these makes experiments hard to reproduce; putting
them all in one flat file becomes unwieldy. Hydra solves this by:

1. **Composing configs from multiple YAML files** — mix and match models, datasets, and trainers
   independently.
2. **Resolving `_target_` strings to Python classes** — so your config *is* your instantiation
   logic (`hydra.utils.instantiate`).
3. **Supporting runtime overrides** — swap any config value from the command line without editing
   files.

---

## Config directory layout

```
conf/
├── config.yaml          ← main config (estimators, paths, downstream tasks)
├── models/
│   ├── my_model.yaml    ← LightningModule config
│   └── other_model.yaml
├── datamodules/
│   └── my_datamodule.yaml
└── trainers/
    └── default.yaml
```

The main `config.yaml` references the group files by filename stem (without `.yaml`). NEEDLE
resolves these references at startup — see [Phase 1 resolution](lightning_and_hydra_integration.md#config-resolution-initialize_hydra_config).

---

## The `MainConfig` dataclass

The top level of the config is validated against `MainConfig`
([`needle/utils/config_schema.py`](../../needle/utils/config_schema.py)):

```python
@dataclass
class MainConfig(SerializableDataclass):
    estimators: dict[str, EstimatorConfig] = field(default_factory=dict)
    downstream_tasks: Optional[dict[str, DownstreamTaskConfig]] = field(default_factory=dict)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    results_path: Optional[str] = None
    results_path_downstream: Optional[str] = None
    custom_settings: Any = None
    _resolved: bool = False
```

| Field | Description |
|---|---|
| `estimators` | Map of estimator name → `EstimatorConfig`. One entry per model to train. |
| `downstream_tasks` | Map of task name → `DownstreamTaskConfig`. User analysis tasks run after training. |
| `aggregation` | How fold/ensemble/systematic results are combined (`mean`, `weighted_mean`, …). |
| `results_path` | Root output directory for training artifacts. |
| `results_path_downstream` | Root output directory for downstream task outputs. Can use OmegaConf interpolations. |
| `custom_settings` | Arbitrary dict for experiment-specific settings. Accessible throughout the config via `${custom_settings.*}`. |

### Minimal `config.yaml`

```yaml
results_path: runs/my_experiment
results_path_downstream: "${results_path}/analysis"

estimators:
  my_estimator:
    model: my_model
    datamodule: my_datamodule
    trainer: default
```

---

## The `EstimatorConfig` dataclass

Each estimator entry is validated against `EstimatorConfig`:

```python
@dataclass
class EstimatorConfig(SerializableDataclass):
    datamodule: str = MISSING
    datamodule_override: Optional[Any] = None
    dataset: Optional[str] = None
    dataset_override: Optional[DatasetConfig] = field(default_factory=DatasetConfig)
    model: str = MISSING
    model_override: Optional[Any] = None
    trainer: str = MISSING
    trainer_override: Optional[Any] = None
    expands: ExpansionConfig = field(default_factory=ExpansionConfig)
    requires: Optional[List[str]] = None
```

### The string + override pattern

Each component (`model`, `datamodule`, `trainer`, `dataset`) follows the same two-field pattern:

- `model: "my_model"` — a string that references `conf/models/my_model.yaml`.
- `model_override: { hidden_dim: 512 }` — per-estimator overrides applied *on top* of the
  referenced file.

After config resolution, the `model` string is no longer present in the resolved config — only
`model_override` is used by `FoldTask`. The resolver populates `model_override` by merging the
named group file with any inline overrides:

```yaml
estimators:
  my_estimator:
    model: my_model             # loads conf/models/my_model.yaml
    model_override:
      hidden_dim: 512           # overrides hidden_dim from the group file
    datamodule: my_datamodule
    trainer: default
```

> **Note:** `dataset` / `dataset_override` merge in the opposite direction — the group config
> takes precedence over inline overrides. This is intentional for dataset configs.

### The `expands` block

`expands` controls how many training tasks are spawned per estimator:

```python
@dataclass
class ExpansionConfig(SerializableDataclass):
    ensembles: EnsembleConfig = field(default_factory=EnsembleConfig)
    systematics: dict[str, SystematicConfig] = field(default_factory=lambda: {"nominal": SystematicConfig()})
    folds: int = 1
```

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
        nominal: {}             # one systematic variation called "nominal"
        high_lr:
          model_override:
            lr: 1e-2            # same model with a different learning rate
```

This config spawns 5 × 3 × 2 = 30 `FoldTask` instances. Each systematic can override any
component (model, datamodule, dataset, trainer) relative to the base estimator config.

### Inter-estimator dependencies (`requires`)

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

---

## Group config files: models, datamodules, trainers

Each group file is a YAML dict that Hydra merges into the `*_override` field of the estimator.
The only required field is `_target_`, which points to the Python class to instantiate.

### Model config (`conf/models/my_model.yaml`)

```yaml
_target_: my_package.models.my_model.MyModel
hidden_dim: 256
lr: 1e-3
```

`_target_` must be a fully-qualified Python dotted path importable from the project root. All
other keys are passed as keyword arguments to the class constructor.

### Datamodule config (`conf/datamodules/my_datamodule.yaml`)

```yaml
_target_: my_package.data.my_datamodule.MyDataModule
batch_size: 512
num_workers: 4
```

NEEDLE passes `dataset_config`, `fold_index`, and `n_folds` as extra kwargs at runtime. If your
datamodule accepts them, it receives them automatically; if not, they are dropped with a warning
(see `hydra_instantiate`).

### Trainer config (`conf/trainers/default.yaml`)

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

---

## The `DatasetConfig` dataclass

Dataset information is validated against `DatasetConfig`:

```python
@dataclass
class DatasetConfig(SerializableDataclass):
    paths: str = ""
    features_columns: Optional[List[str]] = field(default_factory=list)
    labels_columns: Optional[List[str]] = field(default_factory=list)
    format: str = "automatic"
    dak_reader_kwargs: dict[str, Any] = field(default_factory=dict)
    max_number_events: int = -1
```

The dataset can be specified inline in the estimator config (without a group file):

```yaml
estimators:
  my_estimator:
    model: my_model
    datamodule: my_datamodule
    trainer: default
    dataset_override:
      paths: "/data/my_dataset/*.parquet"
      features_columns: ["pt", "eta", "phi", "mass"]
      labels_columns: ["label"]
      max_number_events: 100000
```

Or referenced by name using a group file (`dataset: my_dataset` → `conf/datasets/my_dataset.yaml`).

---

## OmegaConf interpolations

Config values can reference other config fields or environment variables using OmegaConf's
`${...}` syntax:

```yaml
results_path: "runs/my_experiment"
results_path_downstream: "${results_path}/analysis"   # cross-field reference

custom_settings:
  data_dir: "${oc.env:MY_DATA_DIR}"   # reads $MY_DATA_DIR from the environment
  use_test_data: false
  root_dir: ${if:${custom_settings.use_test_data}, "test_data", "${oc.env:MY_DATA_DIR}"}
```

The `if` resolver (`${if:condition, value_if_true, value_if_false}`) is a custom resolver
registered by NEEDLE in `config_utils.py`. It evaluates eagerly at config-load time.

**Caution:** if a required environment variable is not set and `use_test_data` is false, config
loading raises a `MissingMandatoryValue` error immediately — before any task runs.

---

## Downstream task config

Downstream tasks are registered under `downstream_tasks`:

```python
@dataclass
class DownstreamTaskConfig(SerializableDataclass):
    requires: Optional[List[str]] = None
    args: Optional[dict[str, Any]] = field(default_factory=dict)
    expands: Optional[dict[str, Any]] = field(default_factory=dict)
```

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

See [Writing Custom Downstream Tasks](downstream_tasks.md) for the full guide.

---

## Aggregation config

`aggregation` controls how fold, ensemble, and systematic results are combined after training:

```python
@dataclass
class AggregationConfig(SerializableDataclass):
    fold_method: str = "mean"
    ensemble_method: str = "mean"
    systematic_method: str = "mean"
    estimator_method: str = "sum"

    fold_weights: Optional[List[float]] = None
    ensemble_weights: Optional[List[float]] = None
    systematic_weights: Optional[List[float]] = None
    estimator_weights: Optional[List[float]] = None
```

```yaml
aggregation:
  fold_method: "mean"
  ensemble_method: "weighted_mean"
  ensemble_weights: [0.5, 0.3, 0.2]
```

---

## Adding a new estimator

1. **Write your `LightningModule`** inheriting from `lightning.LightningModule`.

2. **Write the group config** in `conf/models/my_model.yaml`:
   ```yaml
   _target_: my_package.models.my_model.MyModel
   hidden_dim: 256
   lr: 1e-3
   ```

3. **Write datamodule and trainer configs** similarly.

4. **Register the estimator** in `config.yaml`:
   ```yaml
   estimators:
     my_estimator:
       model: my_model
       datamodule: my_datamodule
       trainer: default
       dataset_override:
         paths: "/data/*.parquet"
         features_columns: ["feat_a", "feat_b"]
       expands:
         folds: 3
   ```

5. **Run it:**
   ```bash
   law run SnapshotTask \
       --config-file conf/config.yaml
   ```

---

## Config caching

When `MainTask` first runs, it writes the fully resolved config to
`{results_path}/config.yaml`. All downstream tasks in the same run load from this frozen
snapshot. This ensures reproducibility: even if you modify your YAML files mid-run, running
tasks see the original config.

To change the config and rerun, either change `results_path` or manually delete the cached
config file.

---

For a complete working example of a multi-estimator config with systematics and downstream tasks,
see the [FAIR Universe demo](../examples/index.md).
