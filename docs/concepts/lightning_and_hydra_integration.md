# Lightning and Hydra

This page covers how NEEDLE wires together PyTorch Lightning (for training) and Hydra (for
configuration). It describes the internal mechanisms — the `HydraMixin`, `hydra_instantiate`,
and the two-phase config resolution — that every task in the pipeline relies on.

For how to *write* a config file, see [Writing the Configuration](hydra_config.md).

---

## Lightning

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) removes training boilerplate and
provides a fixed interface that makes swapping models or data pipelines easy.

A training run in NEEDLE consists of three objects:

| Class | Responsibility |
|---|---|
| `LightningModule` | Model architecture, optimizer, scheduler, loss |
| `LightningDataModule` | Dataloaders and data preprocessing |
| `Trainer` | Training loop, callbacks, hardware config |

The entry point is always:

```python
trainer.fit(model=model, datamodule=data_module)
```

**Import note:** NEEDLE uses the modern `lightning` package, not the legacy `pytorch_lightning`.
Always import from the modern namespace:

```python
from lightning import LightningModule, LightningDataModule, Trainer
# NOT: from pytorch_lightning import ...
```

Mixing the two causes silent failures where the `Trainer` refuses to accept a `LightningModule`
because they come from different class hierarchies. NEEDLE detects this at instantiation time
and raises a `TypeError` with a clear fix message (see `check_for_lightning_import_mismatch` in
`needle/utils/config_utils.py`).

---

## Hydra and the `HydraMixin`

[Hydra](https://hydra.cc/docs/intro/) is used for configuration management. NEEDLE uses its
`initialize` + `compose` API rather than the `@hydra.main` decorator so it coexists cleanly
with LAW's own argument parser.

```bash
# Correct: Law is the outer layer, hydra operates internally
law run FoldTask --config-file conf/config.yaml --estimator my_estimator

# Avoid: using --hydra.main clashes with Law's argparser
```

### `HydraMixin`

Every task that needs the config inherits from `HydraMixin` (defined in
[`law_tasks/mixins/hydra.py`](../../law_tasks/mixins/hydra.py)). It adds two LAW parameters
and a `config` property:

```python
class HydraMixin:
    config_file: str = law.Parameter(
        description="Path to config folder",
        default="conf/config.yaml",
        significant=True,
    )
    hydra_overrides: str = law.Parameter(
        description="Overrides to pass to Hydra. Format: 'key1=value1 key2=value2'",
        significant=False,
        default="",
    )

    @property
    def config(self) -> MainConfig:
        overrides = self.hydra_overrides.split() if self.hydra_overrides else []
        if hasattr(self, "_config"):
            return self._config
        config_file = Path(str(self.config_file)).resolve()
        self._config = initialize_hydra_config(
            config_dir=str(config_file.parent),
            config_name=str(config_file.stem),
            overrides=overrides,
        )
        return self._config
```

The `config` property is lazy: it only calls `initialize_hydra_config` on first access and
then caches the result on the instance. This means each task process parses the YAML exactly
once, regardless of how many times `self.config` is accessed.

**MRO order:** `HydraMixin` must come before `law.Task` in the class definition:

```python
class MyTask(HydraMixin, law.Task):   # correct
    ...

class MyTask(law.Task, HydraMixin):   # wrong — HydraMixin.config property may be shadowed
    ...
```

### Runtime overrides

Any dot-path in the config can be overridden at the command line via `--hydra-overrides`:

```bash
law run FoldTask \
    --config-file conf/config.yaml \
    --estimator my_estimator \
    --hydra-overrides "estimators.my_estimator.expands.folds=5 results_path=runs/debug"
```

Multiple overrides are space-separated. They are applied *after* the YAML files are loaded.

---

## Config resolution: `initialize_hydra_config`

`initialize_hydra_config` (in [`needle/utils/config_utils.py`](../../needle/utils/config_utils.py))
composes the full config in two phases:

**Phase 1 — string references to full configs (`resolve_defaults`)**

Each estimator entry can reference configs by name, e.g. `model: my_model`. The resolver loads
`conf/models/my_model.yaml` and writes the full dictionary into `model_override`. After this
phase, `config.estimators.my_estimator.model_override` contains the resolved model config dict,
not the string `"my_model"`.

The merge strategy for `*_override` fields is: fields in `model_override` written directly in
the YAML take precedence over fields loaded from the named group file. This lets you set
per-estimator overrides while sharing a base config.

**Phase 2 — OmegaConf interpolation resolution**

`OmegaConf.resolve(cfg)` evaluates all `${...}` expressions (environment variables, cross-field
references, the custom `if` resolver). After this call all values in the config are plain Python
scalars — no deferred references remain.

The fully resolved config is then cast to a `MainConfig` structured dataclass, and
`validate_graph` checks that any `requires` dependencies between estimators are valid.

---

## `hydra_instantiate`: filtered class instantiation

NEEDLE uses a wrapper around `hydra.utils.instantiate` called `hydra_instantiate`
([`needle/utils/config_utils.py`](../../needle/utils/config_utils.py)):

```python
def hydra_instantiate(cfg: DictConfig, **kwargs) -> Any:
    supported_kwargs = {k: v for k, v in kwargs.items() if hydra_check_if_arg_supported(cfg, k)}
    unsupported_kwargs = set(kwargs) - set(supported_kwargs)

    if unsupported_kwargs:
        cls_name = hydra.utils.get_class(cfg._target_).__name__
        logger.warning(
            f"Class {cls_name} does not support the following arguments: "
            f"{unsupported_kwargs}, which were skipped at instantiation."
        )

    return hydra.utils.instantiate(cfg, **supported_kwargs)
```

**Why the wrapper?**

The same `FoldTask.run()` code instantiates *any* model or datamodule class you configure. Some
of those classes may not accept every keyword argument (e.g. `dataset_config` or `input_models`).
Instead of requiring every class to implement the full interface, `hydra_instantiate` inspects
the class signature and silently drops unsupported kwargs, logging a warning. This makes it easy
to write minimal Lightning modules that only accept what they need.

`hydra_check_if_arg_supported` also handles `luigi.Parameter` class attributes, which are not
visible in `__init__` but are valid task parameters.

---

## How `FoldTask` uses all of this

The `run()` method in [`law_tasks/fold.py`](../../law_tasks/fold.py) is the point where the
config, Lightning, and Hydra all come together:

```python
def run(self):
    model_config     = self.systematic_config.model_override
    datamodule_config = self.systematic_config.datamodule_override
    dataset_config   = self.systematic_config.dataset_override
    trainer_config   = self.systematic_config.trainer_override

    # model_config, datamodule_config, trainer_config are resolved DictConfigs with _target_

    model: lightning.LightningModule = hydra_instantiate(
        model_config,
        dataset_config=dataset_config,
        input_models=self.input_model_paths,
    )

    data_module: lightning.LightningDataModule = hydra_instantiate(
        datamodule_config,
        dataset_config=dataset_config,
        input_models=self.input_model_paths,
        fold_index=self.fold_index,
        n_folds=self.estimator_config.expands.folds,
    )

    trainer: lightning.Trainer = hydra.utils.instantiate(
        trainer_config,
        logger=self.mlflow_logger,
    )

    trainer.fit(model=model, datamodule=data_module)
```

`self.systematic_config` is the estimator config merged with any systematic-specific overrides,
so a systematic that only changes one model hyperparameter inherits all other fields from the
base estimator. The `*_override` fields are populated during Phase 1 resolution.

Notice that the `Trainer` is instantiated with plain `hydra.utils.instantiate` (not the wrapper)
because the trainer's accepted arguments are well-known and static.

---

## Writing a Lightning module for NEEDLE

A minimal `LightningModule` that works with `hydra_instantiate`:

```python
import lightning as L

class MyModel(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        lr: float,
        dataset_config: dict,   # injected by hydra_instantiate if accepted
        input_models: dict,     # injected if present; dropped silently if not
    ) -> None:
        super().__init__()
        self.lr = lr
        self.net = ...

    def training_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

The corresponding YAML in `conf/models/my_model.yaml`:

```yaml
_target_: my_package.models.my_model.MyModel
hidden_dim: 256
lr: 1e-3
```

`dataset_config` and `input_models` are injected at task runtime and do not need to appear in
the YAML — they are passed as extra `kwargs` to `hydra_instantiate`.
