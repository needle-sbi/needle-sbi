# Lightning and Hydra

Using pytorch barebones yields a lot of control over the individual training loops but also
complicates the interface between law and the training. At some point, have a function or class
that handles the training that can be easily called by law is desirable for separation and
readability.

We use torch lightning to facilitate this interface and benefit from the reduced boilerplate code
necessary to set up new models.

## Lightning

https://lightning.ai/docs/pytorch/stable/

This is a extensive wrapper around pytorch that removes a lot of boilerplate such as setting up
the training loop or handling epochs. It still provides many callbacks that allows one to connect
at desired times during training. Most importantly, lightning modules have a fixed interface that
allows one to exchange them easily.

A lightning training consists of three main ingredient:
 - `LightningModule`: Configuration of your models, optimizer, schedulers and loss functions.
 - `LightningDataModule`: Configuration of the dataloaders and how to obtain the data. Can be
    used to exchange dataformats, e.g. if we want to shift from awkward to pandas we simply
    write a new DataModule and select that one instead of the old one.
 - `Trainer`: The default Lightning one already most of the things and defines how many epochs
    with what callbacks the training is to be run.

In our Law Tasks we use the following interface:

```python
model: lightning.LightningModule
data_module: lightning.LightningDataModule
trainer: lightning.Trainer

trainer.fit(model=model, datamodule=data_module)
```

### Typed schema layer for module-specific validation

Each instantiable class (model, datamodule, trainer) has a corresponding `@dataclass` schema registered with Hydra's `ConfigStore` under the appropriate config group. This gives compose-time validation so typos and type errors can be caught before any code actually runs. This also allows for polymorphic config groups, such that adding a new model only requires a new schema (just a call to `cs.store()`) and a YAML file. The actual dispatch logic remains the same. If a config is built instead from a dictionary, e.g. bypassing a compose step from Hydra, then dedicated helpers are provided in the api to check fields against the schema. Unknown `_target_` values pass through unchanged, which should preserve extensibility. An example useage is shown below:

```python
validated_model_cfg = validate_model_config(self.config.models)
validated_data_cfg = validate_datamodule_config(self.config.datamodules)
validated_trainer_cfg = validate_trainer_config(self.config.trainers)

model = hydra.utils.instantiate(
    validated_model_cfg,
    dataset_config=self.config.datasets,
)
data_module = hydra.utils.instantiate(
    validated_data_cfg,
    dataset_config=self.config.datasets,
)
trainer = hydra.utils.instantiate(
    validated_trainer_cfg,
    logger=mlflow_logger,
)
```

How modules are loaded and orchestrated is covered in the next section.

## Hydra

https://hydra.cc/docs/intro/

This is used mainly to sort out our configuration, which we expect to be very large once we add many
models to the codebase. Hydra uses a main config set in `conf/config.yaml` by default that can
import config settings from sub-folders in the same directory. Hydra then returns a single config
that can be accessed as a dict (even with attribute-based access).

For example, setting the number of folds is a float in the main `config.yaml`. This can be easily
accessed with `config.n_folds`. If we nest configs then we might end up with `conf/trainers/default.yaml`
that contains `max_epochs = 10`. Now we access this with `config.trainers.max_epochs` inside our
config.

### CLI with Hydra and Law

Hydra allows one to change all the config parameters at runtime from the CLI using the integrated
argparser when using the `@hydra.main` decorator.

```bash
python3 my_script.py --trainers.max_epochs=5
```

However, when using Law (we are), the two argparsers can clash. For best compatibility:

```bash
python3 my_script.py <hydra_config_args> -- <law_parameters>
```

The arguments have to be split using the empty flag `--`. Unfortunately, we lose the tab autocompletion
from Law in this case.

Best solution: Keep Law as the outer layer and do now use hydra in the CLI at all

```
law run TrainingBaseTask
``` 

This way we only use hydra with the `initialize` and `compose` functions, not the `@hydra.main` decorator.

### Hydra for instantiating classes

One very nice thing that hydra allows is to automatically call functions or instantiate classes from
a given config. This ties in neatly with lightning as one can simply exchange lightning modules by
changing the config entry.

For example, in our law Task we might call:

```python
model: lightning.LightningModule = hydra.utils.instantiate(
    self.config.models,
    dataset_config=self.config.datasets,
)
data_module: lightning.LightningDataModule = hydra.utils.instantiate(
    self.config.datamodules,
    dataset_config=self.config.datasets,
)
trainer: lightning.Trainer = hydra.utils.instantiate(self.config.trainers)

trainer.fit(model=model, datamodule=data_module)
```

It would be even nicer if we could abstract away the config for the datasets too but this seems difficult
since the datasets include info about which columns to load. Doing it this way streamlines the passing
of configs a lot.

A lightning Module might then have this shape:

```python
class MockTransformerModule(L.LightningModule):
    def __init__(
        self,
        factor: float,
        patience: int,
        init_lr: float,
        dataset_config: dict,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.num_features = len(DatasetConfig(**dataset_config).features_columns)
        self.init_lr = init_lr
```

Despite the lightning DataModule handling all the data-related code, it cannot be avoided that some
models require information about the shape of data at runtime. We could use hydra tools to link
`num_features` to the list of columns in the `conf/datasets/` config, for example with a custom
implementation of `$(len(feature_columns))` but whether this works well remains to be seen.