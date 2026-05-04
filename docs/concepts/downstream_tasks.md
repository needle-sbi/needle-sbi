# Writing Custom Downstream Tasks

Downstream tasks are how you attach your own analysis code to the NEEDLE pipeline. After all
models are trained and the snapshot is written, `DownstreamTask` instantiates and runs whatever
Luigi `Task` you specify in the config.

## Anatomy of a downstream task class

A downstream task is just a `luigi.Task`. NEEDLE does not impose any special base class.

```python
import luigi

class MyAnalysisTask(luigi.Task):
    # LAW/Luigi parameters become constructor arguments
    snapshot_path: str = luigi.Parameter(description="Path to the DAG snapshot JSON")
    output_path: str = luigi.Parameter(description="Where to write results")
    root_dir: str = luigi.Parameter(description="Path to raw data")

    def output(self):
        # Tell Luigi which files this task creates
        return luigi.LocalTarget(self.output_path)

    def run(self):
        # Your analysis code goes here
        import json
        with open(self.snapshot_path) as f:
            snapshot = json.load(f)

        # ... do something with the trained models ...

        with open(self.output_path, "w") as f:
            json.dump({"result": 42}, f)
```

Key rules:
- `output()` must return a `LocalTarget` (or dict of them) whose paths will be created by `run()`.
- Luigi checks `output()` to decide if the task is already done. If all output files exist, the
  task is skipped.
- The `run()` method must create all output files before it exits. If it raises an exception,
  the task is marked failed and downstream tasks will not run.

## Registering it in the config

Add an entry to `downstream_tasks` in your config YAML:

```yaml
downstream_tasks:
  my_analysis:
    requires: ["histogram"]    # optional: wait for these other downstream tasks first
    args:
      _target_: my_package.tasks.my_task.MyAnalysisTask
      root_dir: "${custom_settings.root_dir}"
      output_path: "${results_path_downstream}/my_results.json"
```

The `snapshot_path` parameter is injected automatically by `DownstreamTask` — you do not need
to specify it in the config.

OmegaConf interpolations (`${...}`) are resolved before the task class is instantiated, so
`root_dir` will have the actual path string by the time `MyAnalysisTask.__init__` is called.

## Running it

```bash
law run DownstreamTask \
    --downstream my_analysis \
    --config-file conf/config.yaml
```

LAW will:
1. Run `SnapshotTask` (and therefore the entire training pipeline) if not already complete.
2. Run any tasks listed in `requires` if not already complete.
3. Instantiate and run `MyAnalysisTask`.

## Parameter expansion

If you want to run the same downstream task with different parameter values (e.g. validate each
model variant separately), use `expands`:

```yaml
downstream_tasks:
  validate_nf:
    args:
      _target_: my_package.tasks.ValidateNF
      root_dir: "${custom_settings.root_dir}"
    expands:
      model_name: ["nf_signal_1jet", "nf_background_1jet", "nf_signal_2jet"]
```

NEEDLE spawns one `DownstreamTask` per value in the expanded list. Each gets the extra parameter
passed to the task constructor:

```python
class ValidateNF(luigi.Task):
    model_name: str = luigi.Parameter()
    snapshot_path: str = luigi.Parameter()
    root_dir: str = luigi.Parameter()
    ...
```

## Accessing trained models from the snapshot

The snapshot JSON has the following structure. Parse it to find checkpoint paths:

```python
import json
from urllib.parse import parse_qs

with open(snapshot_path) as f:
    snapshot = json.load(f)

for node_name, node in snapshot["nodes"].items():
    params = parse_qs(node_name)          # parses "est=nf_signal_1jet&syst=c_0p5&..."
    estimator = params["est"][0]
    ckpt_path = node["checkpoint_path"]
    # Load with Lightning:
    model = MyModel.load_from_checkpoint(ckpt_path)
```

The FAIR Universe demo's `HistogramTask.parse_snapshot()` is a good reference implementation.

## Using `cached_property` for expensive setup

If your task loads large data or models that should not be reloaded on every method call,
use `functools.cached_property`:

```python
from functools import cached_property

class MyTask(luigi.Task):
    root_dir: str = luigi.Parameter()

    @cached_property
    def loaded_data(self):
        return load_my_data(self.root_dir)  # called once, cached

    def run(self):
        data = self.loaded_data  # fast after first call
```

## Chaining downstream tasks

The `requires` key creates ordered dependencies between downstream tasks:

```yaml
downstream_tasks:
  histogram:
    args: { ... }
  neyman:
    requires: ["histogram"]
    args: { ... }
  eval:
    requires: ["neyman"]
    args: { ... }
```

When you run `law run DownstreamTask --downstream eval`, LAW will run `histogram` then `neyman`
then `eval`, checking output file existence to skip already-complete steps.

## Plotting utilities (`PlottingMixin`)

The demo provides `PlottingMixin` in `tasks/plotting_mixin.py` as a convenient base for tasks
that produce plots. It provides a `@plot` decorator that:
- Automatically creates the output directory.
- Registers the plot file as a task output.
- Calls `fig.savefig(...)` for you.

```python
from .plotting_mixin import PlottingMixin

class MyTask(PlottingMixin):
    plot_save_dir: str = luigi.Parameter()

    @PlottingMixin.plot(name="my_plot")
    def make_plot(self, data) -> Figure:
        fig, ax = plt.subplots()
        ax.plot(data)
        return fig  # PlottingMixin saves it

    def run(self):
        self.make_plot(some_data)
```
