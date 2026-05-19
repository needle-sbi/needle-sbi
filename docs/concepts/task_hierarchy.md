# DAG Workflow

## What is a DAG and why does NEEDLE use one?

A **Directed Acyclic Graph (DAG)** is a set of nodes connected by directed edges, with no cycles.
In NEEDLE, each node is a LAW Task (a Python class) and each edge is a dependency: task B depends
on task A meaning A must finish before B can start.

This pattern is standard in HEP analysis (ColumnFlow, Luigi, CRAB all use it) because:
- It makes the dependency structure explicit and auditable.
- Partially complete runs can be resumed: LAW re-checks which outputs exist and only re-runs
  what is missing.
- The same task graph can be executed locally, on SLURM, or on HTCondor by swapping a mixin.

## The full task hierarchy

```
MainTask (WrapperTask — no output, just coordinates)
 └── EstimatorTask (one per estimator, e.g. "nf_signal_1jet")
      └── SystematicTask (one per systematic variation, e.g. "c_0.5")
           └── EnsembleTask (repeated N times for ensemble members)
                └── FoldTask (one per cross-validation fold)
                     └─── [PyTorch Lightning trainer.fit(...)]

SnapshotTask
 └── requires MainTask
 └── writes dag_snapshot.json with all checkpoint paths

DownstreamTask
 └── requires SnapshotTask + declared upstream downstream tasks
 └── instantiates user-provided task class via Hydra
```

For the FAIR Universe demo this expands to approximately:

```
MainTask
 ├── EstimatorTask(nf_signal_1jet)
 │    ├── SystematicTask(c_0.5) → EnsembleTask → FoldTask → training
 │    └── SystematicTask(c_2.0) → EnsembleTask → FoldTask → training
 ├── EstimatorTask(nf_signal_2jet)    [same pattern]
 ├── EstimatorTask(nf_background_1jet) [same pattern]
 ├── EstimatorTask(nf_background_2jet) [same pattern]
 └── EstimatorTask(classifier)
      └── SystematicTask(nominal) → EnsembleTask → FoldTask → training
           (waits for all four NF estimators to finish first)
```

## What each task does

### `FoldTask` — the leaf node

This is where the actual training happens. Each `FoldTask` instance is parameterised by:
- `estimator` — which estimator configuration to use
- `systematic` — which systematic variation
- `ensemble` — ensemble member index
- `fold_index` — cross-validation fold index

The `run()` method:
1. Loads the resolved `MainConfig` from the config file.
2. Instantiates the `LightningModule` (model), `LightningDataModule`, and `Trainer` via Hydra.
3. Calls `trainer.fit(model, datamodule)`.
4. Saves the checkpoint, metrics JSON, model config YAML, and training logs to its output directory.

The output directory path encodes all parameters:
```
{results_path}/est__{estimator}/syst__{systematic}/ensem__{ensemble}/fold__{fold_index}/
```

LAW considers `FoldTask` complete when all output files exist. If training crashes mid-way, LAW
will re-run the full task on the next invocation (no partial checkpoint loading).

### `EnsembleTask`

Requires `N` copies of `FoldTask` (where N = `expands.ensembles.num_ensembles`) and aggregates
their `FoldResults` into an `EnsembleResults` object. The aggregation method (e.g. `mean`,
`weighted_mean`, `best`) is configured per estimator.

Having multiple ensemble members is useful for uncertainty estimation: you train the same model
multiple times with different random seeds and use the spread of outputs as an epistemic
uncertainty estimate.

### `SystematicTask`

Requires one `EnsembleTask` per systematic variation. The systematics are variations of model and the
data. Both can be steered for each Systematics by using the `*_override` field in the config to swap
out model or data. For example, training on two different hyperparameters will require two Systematics.
Training on nominal, up and down variations for a given physics systematics will require three
SystematicTask

After all ensembles finish, `SystematicTask` aggregates results into `SystematicResults`.

### `EstimatorTask`

One per top-level entry in `config.estimators`. Requires all `SystematicTask` instances for that
estimator and produces an `EstimatorResults`. It also handles the special case where `requires`
is set (see inter-estimator dependencies below).

### `MainTask`

A `WrapperTask` — it has no output file of its own and never marks itself as complete. Its sole
job is to require all `EstimatorTask` instances. You should not run `MainTask` directly;
instead run `SnapshotTask` which requires `MainTask`. You could run `MainTask` in case you are just
interested in training the models but do not want to perform inference afterwards.

### `SnapshotTask`

Walks the output tree of `MainTask` and collects all checkpoint paths into a single JSON file
(`dag_snapshot.json`). This file is the handshake between the training pipeline and the
downstream analysis: downstream tasks load models from this file without needing to know how
training was organised.

The snapshot format:
```json
{
  "nodes": {
    "est=nf_signal_1jet&syst=c_0.5&ensem=0&fold=0": {
      "checkpoint_path": "runs/.../best.ckpt",
      "task_type": "fold",
      "fold_index": 0,
      "ensemble_index": 0,
      "estimator_name": "nf_signal_1jet",
      "systematic_name": "c_0.5",
      "metrics": { "val_loss": 0.42 }
    }
  },
  "edges": [ ... ]
}
```

### `DownstreamTask`

A generic wrapper for user-defined analysis tasks. It:
1. Requires `SnapshotTask` and any declared upstream `downstream_tasks` keys.
2. Reads the config to find the task's `_target_` class and `args`.
3. Instantiates the user task via `hydra_instantiate` (so you can use `_target_` in your config).
4. Calls the task's `run()` method.

The user task is a plain Luigi `Task` — it does not need to know about LAW or Hydra.

## Inter-estimator dependencies (`requires` field)

Some estimators need outputs from other estimators before they can train. In the FAIR Universe
demo, the `classifier` requires all four normalizing flow estimators:

```yaml
estimators:
  classifier:
    requires:
      - nf_background_1jet
      - nf_background_2jet
      - nf_signal_1jet
      - nf_signal_2jet
    model: classifier
    ...
```

At runtime, `EstimatorTask(classifier)` will not start until all four NF `EstimatorTask`
instances have completed. The classifier's `FoldTask` then loads the trained NF checkpoints
(passed through the snapshot mechanism) to compute NF-based features during training.

This is how you build a two-stage pipeline: first train auxiliary models, then train a final
model that uses their outputs as features.

## Expansion: systematics and ensembles

The `expands` block in each estimator config controls how many tasks get spawned:

```yaml
estimators:
  nf_signal_1jet:
    expands:
      systematics:
        c_0.5:
          model_override: { c: 0.5 }
        c_2.0:
          model_override: { c: 2.0 }
      ensembles:
        num_ensembles: 1
      folds: 1
```

NEEDLE will create:
- 1 `EstimatorTask`
- 2 `SystematicTask` instances (one per systematic key)
- 1 `EnsembleTask` per systematic (num_ensembles=1)
- 1 `FoldTask` per ensemble (folds=1)

Total: 2 training runs for this estimator. If you set `num_ensembles: 3` and `folds: 5` you
would get 2 × 3 × 5 = 30 training runs.

## Why separate systematics from the physics systematics?

The `expands.systematics` in the NEEDLE config controls **model-level variations** — things like
different hyperparameters or different training data subsets. The physics systematics (JES, TES,
etc.) are handled inside the data pipeline during inference, not during training.

This separation is intentional: you train a set of models, then at inference time you apply
physics-level systematic shifts to the data and feed them through the (fixed) trained models.
The model-level systematics in `expands` let you train multiple variants of the same model to
study sensitivity to training choices.

## Adding a new task to the pipeline

To add a new downstream analysis task:

1. Write a class inheriting from `luigi.Task` in your example package.
2. Add it to `downstream_tasks` in your config YAML:
   ```yaml
   downstream_tasks:
     my_analysis:
       requires: ["histogram"]   # wait for histogram task
       args:
         _target_: my_package.tasks.my_task.MyTask
         some_param: "value"
   ```
3. Run it:
   ```bash
   law run DownstreamTask --downstream my_analysis \
       --config-file conf/config.yaml
   ```

Your task will receive `snapshot_path` automatically (injected by `DownstreamTask`) plus
any `args` from the config.
