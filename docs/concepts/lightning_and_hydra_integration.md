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

::: {important} 
NEEDLE uses the modern `lightning` package, not the legacy `pytorch_lightning`.
Always import from the modern namespace:

```python
from lightning import LightningModule, LightningDataModule, Trainer
# NOT: from pytorch_lightning import ...
```

Mixing the two causes silent failures where the `Trainer` refuses to accept a `LightningModule`
because they come from different class hierarchies. NEEDLE will detect this at instantiation time
and raises a `TypeError` with clear instruction on how to migrate to newer Lightning.
:::

## Writing a Lightning module for NEEDLE

A minimal `LightningModule` that works with `hydra_instantiate`:

```python
import lightning as L

class MyModel(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        lr: float,
        dataset_config: dict,   # <- optional args injected by hydra
        input_models: dict,     # <-
    ) -> None:
        super().__init__()
        self.lr = lr
        self.model = ...

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

NEEDLE provides additional arguments like `dataset_config` and `input_models` which are injected
at task runtime by the framework if your LightningModule accepts them. If you dont need these inputs,
simply do not add them to the `__init__` and hydra will drop them without errors.

 - `dataset_config` is an additional (optional) config group if using the NEEDLE built-in LightningDatamodules.
 - `input_models` provides you with a dictionary view on all the models referenced by the `requires` 
    keyword for the corresponding estimator. Meaning if model B depends on A, you can access the path to the model checkpoint from model A using this dictionary.

## Building the config

NEEDLE has the same three stage config build as regular Hydra. You can find much more information
on how Hydra composes the config from the [Hydra docs](https://hydra.cc/docs/intro/).

### Step 1: References to other config files

Writing down all the settings for many individual models can become tedious and unorganized if
everything is in a single file. Hydra allows you to instead pass a reference to config files
by using the string name of the file. These sub-config files must live in the same directory,
for example `conf/` (the naming is not strictly enforced):

```
conf/                   # main config folder
    config.yaml         # main config file
    models/             # group folder
        my_model.yaml   # sub-config file
    datasets/
    datamodules/
    trainers/
```

Each estimator entry can reference configs by name, e.g. `model: my_model`. The resolver loads
`conf/models/my_model.yaml` and writes the content of the file into the dictionary into the
main config.

::: {admonition} Difference between `models` folder and `model` field
:class: hint
We found it logical to use plural for the group folders `models`, `datasets`, `datamodules` and `trainers` and use the singular for the group name `model` and `model_override`. This is more natural since `models/` are supposed to hold multiple individual sub-configs, each representing
    a single model. Hydra by default would have used the same name for both.
:::

::: {admonition} How are the fields populated?
:class: note

Using the config structure shown above, we have a sub-config `models/my_model.yaml` with the
following hyperparameters:

    _target_: my_package.classifier.py
    lr: 0.001
    hidden_dim: 128
    latend_dim: 128

What the fields actually mean is explained in [Writing the Configuration](hydra_config.md). Here
it is only important that we have some fields with some values (string, float, int...).

Inside the main `conf/config.yaml`, we simply reference our sub-file by name, without the .yaml ending:

```yaml
estimators:
    my_estimator:
        model: my_model
```

Now, Hydra will fetch the content of the file and add them to a new field `model_override`.

```yaml
estimators:
    my_estimator:
        model: my_model  # unchanged
        model_override:
            _target_: my_package.classifier.py
            lr: 0.001
            hidden_dim: 128
            latend_dim: 128
```

The group name in the config is kept the same, since it allows you to track where the resolved config entries came from. These are the values that are then passed on to your Lightning modules.
:::

::: {admonition} Manual overrides in `config.yaml`
:class: note
If you want to adjust one or more config entries after the group resolution, simply add the changes to
the `*_override` field before. By default it is empty and only populated during group resolution,
but you can add it yourself for full control:

```yaml
estimators:
    my_estimator:
        model: my_model
        model_override:         # add this yourself
            hidden_dim: 256     # change one or more values
```

Which produces the resolved config:

```yaml
estimators:
    my_estimator:
        model: my_model
        model_override:
            _target_: my_package.classifier.py
            lr: 0.001
            hidden_dim: 256  # manual override
            latend_dim: 128
```


Fields written directly in `*override` take precedence over fields loaded from the sub-config. This 
lets you set per-estimator overrides while sharing a base config.
:::

### Step 2: OmegaConf interpolation resolution

Hydra builds upon [OmegaConf](https://github.com/omry/omegaconf) which has many powerful features
for writing configs. This includes:

 - Interpolation strings like `${foo.bar}` which allows you to access variables defined elsewhere
    in the config.
 - If statements with `${if:<condition>, <value_if_true>, <value_if_false>}"`. This is a custom
    NEEDLE resolver.
 - Environment variables with `${oc.env:FAIR_UNIVERSE_DATA,""}` which is equivalent to python
    `os.getenv()`.

For more information see the [OmegaConf docs](https://omegaconf.readthedocs.io/en/latest/).

### Step 3: Runtime overrides

Any dot-path in the config can be overridden at the command line via `--hydra-overrides`:

```bash
law run MainTask \
    --config-file conf/config.yaml \
    --hydra-overrides "estimators.my_estimator.model_override.hidden_dim=512"
```

Multiple overrides are space-separated. Changes here again take precedence of previous overrides.

::: {admonition} Override order
:class: attention
Runtime overrides > Manual override in `config.yaml` > Field value in sub-config
:::

### Schema and DAG validation

After building the config, NEEDLE validates the final, resolved config using OmegaConf. This
is where `TypeErrors` will occur if the entry for a given field does not match the expected
type in the python strucured dataclass for `needle.utils.MainConfig`.
Finally, the DAG defined in the config is fed to a topological solver that checks that all
the inter-estimator references from the `requires` keyword are not circular and can be passed to the DAG workflow.

## What happens under the hood?

The remainder of this page is a more technical explanation aimed at experts.

### `hydra_instantiate`: filtered class instantiation

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

## Hydra under the hood

[Hydra](https://hydra.cc/docs/intro/) is used for configuration management. NEEDLE uses its
`initialize` + `compose` API rather than the `@hydra.main` decorator so it coexists cleanly
with LAW's own argument parser.

```bash
# Correct: Law is the outer layer, hydra operates internally
law run FoldTask --config-file conf/config.yaml --estimator my_estimator

# Avoid: using --hydra.main clashes with Law's argparser
```

### `HydraMixin`

Every task that needs the config inherits from `HydraParamsMixin` (defined in
[`needle/tasks/mixins/hydra.py`](../../needle/tasks/mixins/hydra.py), shared by both
backends). It adds two parameters and a `config` property:

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

## How `TrainingTask` uses all of this

The `run()` method in [`needle/tasks/base/training.py`](../../needle/tasks/base/training.py)
(`BaseTrainingTask`, shared by both backends — `FoldTask` is just its `.done`-marker parent)
is the point where the config, Lightning, and Hydra all come together:

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
