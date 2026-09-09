# Writing Custom Downstream Tasks

Downstream tasks allow you to attach your own analysis code to the NEEDLE pipeline. After all
models are trained and the snapshot is written, `DownstreamTask` instantiates and runs whatever
Luigi `Task` you specify in the config.

::: {admonition} When to use `DownstreamTask`
:class: tip
`DownstreamTask` exists so you can point NEEDLE to your Task purely through config. If you already have a `luigi`/`law`/`b2luigi` workflow or plan to build
one then importing `needle-sbi` as a package is simpler than using `DownstreamTask`. In this case,
you only need to point your own `requires()` method to NEEDLE's `MainTask` directly from your own
workflow, see the [Scenarios](#scenarios) section below.
:::

::: {admonition} Why is my luigi Task not shown in the DAG?
:class: info
When using the DownstreamTask wrapper for your luigi Task, it is not possible for luigi or the other
schedulers to inspect your Task. This is because it would have to read the config file first in order
to see which DownstreamTasks are registered, but luigi only performs a static analysis.

Therefore, `DownstreamTask` cannot simply `require()` your Task the normal luigi way: luigi
would need to know its class and parameters *before* the config is parsed. Instead,
`DownstreamTask` reads the config first, then instantiates and drives your Task directly
(`output()`/`run()`/`complete()`), which is why it shows up in the DAG only as a generic
`DownstreamTask` node rather than as your own Task class.
:::

(scenarios)=
## Scenarios

There are two ways to combine your own tasks with NEEDLE's:

- **NEEDLE-in-yours**: Import NEEDLE's `MainTask` (law or b2luigi backend) and `require()`
  it from your own `luigi`/`law`/`b2luigi` task, running everything through your own scheduler.
- **Yours-in-NEEDLE**: Keep your task in your own code and register it under
  `downstream_tasks` in the config. NEEDLE's `DownstreamTask` instantiates and runs it for you
  (this page's main topic).

## Compatibility matrix

|                       | plain `luigi`             | `law`                     | `b2luigi`             |
|-----------------------|---------------------------|---------------------------|-----------------------|
| **NEEDLE-in-Yours**   |  Not supported (breaks batch submissions) | ☑ `needle.tasks.law`| ☑ `needle.tasks.b2luigi` |
| **Yours-in-NEEDLE** (using `DownstreamTask`) |        ☑ |                         ☑ |                     ☑ |

Note: `DownstreamTask` calls the wrapped task's `output()`/`run()`/`complete()` directly rather than
scheduling it, so the wrapped class only needs to *look* like a `luigi.Task`. Therefore, a plain `luigi.Task`, a `law.Task`, or a `b2luigi.Task` all work with
either backend's `DownstreamTask`.

::: {warning}
The `requires()` method of your `luigi`/`law`/`b2luigi` Task cannot be to define the dependency graph of your post-training with `DownstreamTask`. Instead, use the `requires`
section in the config.

Wrong: Using the `require()` method in your Luigi Task:
```python
class MyAnalysisTask(luigi.Task):
    def requires(self):  # Will be silently ignored.
        return OtherAnalysisTask()

    def run(self):
        ...
```

Correct: Declare the dependency in the config instead, see [Chaining_Downstream_Tasks](#chaining-downstream-tasks)

```yaml
downstream_tasks:
  other_analysis:
    args: { _target_: my_package.tasks.OtherAnalysisTask }
  my_analysis:
    requires: ["other_analysis"]
    args: { _target_: my_package.tasks.MyAnalysisTask }
```
:::

## Implementing a DownstreamTask

A downstream task is just a `luigi.Task`. NEEDLE does not impose any special base class.

```python
import luigi
import json


class MyAnalysisTask(luigi.Task):
    snapshot_path: str = luigi.Parameter()  # optional, see next Section
    output_path: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        with open(self.snapshot_path) as f:
            snapshot = json.load(f)

        # ... do something with the trained models ...

        with open(self.output_path, "w") as f:
            json.dump({"result": 42}, f)
```

Key rules:
- `output()` must return a `LocalTarget` (or a collection of them) whose paths will be created by
    `run()`. Acceptable collections when running with `--backend law` are `list`, `dict` and `tuple`.
- Luigi checks `output()` to decide if the task is already done. If all output files exist, the
  task is skipped.
- The `run()` method must create all output files before it exits. If it raises an exception,
  the task is marked as failed.

::: {hint}
Ensure that the content of the files at the end of the `run()` method are correct. Otherwise, if the
file is corrupted or missing content, luigi will consider the Task as done. In that case, you might get
a more cryptic error downstream that is more difficult to trace back to the corrupted file.
:::

## Registering a new DownstreamTask

Add an entry to `downstream_tasks` in your config YAML:

```yaml
downstream_tasks:
  my_analysis:                  # name for this step
    requires: ["histogram"]     # optional: wait for these other downstream tasks first
    args:
      _target_: my_package.tasks.my_task.MyAnalysisTask
      root_dir: "${custom_settings.root_dir}"
      output_path: "${results_path_downstream}/my_results.json"
```

The `downstream_tasks` field is a dict in the same style as `estimators`. You can register an arbitrary
number of DownstreamTasks (here just one named `my_analysis`). The valid sub-fields for each entry are:

| Field                   | Python Type                       | Description                       |
|-------------------------|-----------------------------------|-----------------------------------|
| `requires`              | `Optional[List[str]]`             | Reference other entries by name    |
| `args`                  | `dict[Any]`                       | Required. Needs at least the `_target_` field as an entry. All the other args are passed to your Task. See [Schema for the `args` entry](#the-args-target-structure) |
| `expands`               | `Optional[dict[str, list[Any]]]`  | How to duplicate this task. Use a descriptive name for each key and use a list of values to iterate over. If passing more than one key-value, then the cartesian product of those keys are used. See [expands block](./downstream_tasks.md#parameter-expansion) |

::: {admonition} The `snapshot_path` parameter
:class: note
This parameter is injected automatically by `DownstreamTask` and points to the path of the snapshot of the trained models.
While you do not need to specify it in the config, you have to accept it as an argument in your Task if you want to access it. It is completely optional, so if your Task *does not* accept it, then
it will be dropped with an info message. This parameter exists to automatically track the
location of the training output directory in your DownstreamTasks. This uses the same
kwarg-injection mechanism as `model`/`datamodule` classes (See
[Runtime-injected arguments](hydra_config.md#runtime-injected-arguments) for the full picture).
:::

::: {hint}
OmegaConf interpolations (`${...}`) are resolved before the task class is instantiated, so
`root_dir` will have the actual path string by the time `MyAnalysisTask.__init__` is called.
:::

(the-args-target-structure)=
## Schema for the `args` entry

Unlike `estimators`, the entries for `downstream_tasks.<my_analysis>` are freeform dicts. NEEDLE cannot know or validate the shape of your Task ahead of time. The only requirement is the Hydra `_target_` key that point to the python module with your Task.

For example:

```yaml
downstream_tasks:
  my_analysis:
    args:
      _target_: my_package.tasks.my_task.MyAnalysisTask
      root_dir: "${custom_settings.root_dir}"
      output_path: "${results_path_downstream}/my_results.json"
```

This gets unpacked in three steps:

```python
# 1. args, with OmegaConf interpolations resolved to plain Python values
base_args = OmegaConf.to_container(self.downstream_config.args, resolve=True)
# base_args = {
#   "_target_": "my_package.tasks.my_task.MyAnalysisTask",
#   "root_dir": "/abs/path/from/custom_settings.root_dir",
#   "output_path": "runs/default/analysis/my_results.json"
# }

# 2. The `expands` values for this specific branch are merged in on top
# (see Parameter expansion below)
merged_args = {**base_args, **branch_args}

# 3. hydra_instantiate resolves `_target_` to a class and calls it with the filtered kwargs
return hydra_instantiate(merged_args, snapshot_path=self.snapshot_path)
```

What this means concretely:

- Every other key in `args` must match a `luigi.Parameter` name your Task declares
  (`root_dir`, `output_path` in the example above). An `args` key with no matching parameter on the
  Task raises a normal `TypeError: unexpected keyword argument` from luigi's own parameter
  resolution.
- `snapshot_path` is the one extra kwarg NEEDLE injects on every call regardless of what's in
  `args`. If your Task does not declare a `snapshot_path` luigi parameter, it is dropped with a
  warning. This preferential treatment only applies to `snapshot_path` (being a `kwarg` and not 
  part of the Config dict).
- Values are passed as-is after OmegaConf parsing. Make sure your Task's `luigi.Parameter` type   
  matches what the config actually produces.

## Running your DownstreamTasks

From `law`:

```bash
law run DownstreamTask --downstream my_analysis
```

From `needle` (as a positional argument):

```bash
needle run DownstreamTask my_analysis
```

Or explicitly:

```bash
needle run DownstreamTask --param downstream=my_analysis
```

Either way:
1. `MainTask` (and therefore the entire training pipeline) runs first if not already complete.
2. Any tasks listed in `requires` run if not already complete.
3. `MyAnalysisTask` is instantiated and run.

See [DAG Workflow](task_hierarchy.md) for how the two backends differ.

(parameter-expansion)=
## Parameter expansion

If you want to run the same downstream task with different parameter values (e.g. validate each
model variant separately), use `expands`. This is the downstream-task equivalent of the
estimator-level [`expands` block](hydra_config.md#the-expands-block), but simpler: instead of the
fixed `folds`/`ensembles`/`systematics` schema, `expands` here is an arbitrary
`dict[str, list[Any]]`. Any key you name becomes a keyword argument on your Task, as long as your
Task declares a matching `luigi.Parameter`.

For example, lets create 3 duplicates of the `validate_nf` DownstreamTask:

```yaml
downstream_tasks:
  validate_nf:  # mirroring the example/fair_universe_demo config
    args:
      _target_: my_package.tasks.ValidateNF
      root_dir: "${custom_settings.root_dir}"
    expands:
      model_name: ["nf_signal_1jet", "nf_background_1jet", "nf_signal_2jet"]
```

This spawns one `DownstreamTask` instance per value in the expanded list, one per `model_name`.
For each branch, the single `{model_name: "nf_signal_1jet"}` pair is merged on top of
`args` and both are handed to `hydra_instantiate` (see
[Schema for the args entry](#the-args-target-structure) above) as keyword arguments, so your
Task must declare the matching parameter:

```python
class ValidateNF(luigi.Task):
    model_name: str = luigi.Parameter()   # <- filled from `expands.model_name` per branch
    root_dir: str = luigi.Parameter()
```

When `expands` lists multiple keys, it will spawn one branch per combination of the **cartesian
product** of all value lists (via `itertools.product`).

For example, if we want 2 × 2 branches:

```yaml
expands:
  model_name: ["nf_signal_1jet", "nf_background_1jet"]
  jet_bin: ["1jet", "2jet"]
```

This will create four duplicates, the first instance being 

```
DownstreamTask(downstream="validate_nf", model_name="nf_signal_1jet", jet_bin="1jet")
```

Of course, the `ValidateNF` Task above would then also need a new `jet_bin: str = luigi.Parameter()` field.

::: {hint}
The plain naming for each branch is determined using `urlencode(sorted(branch_params.items()))`, e.g. `jet_bin=1jet&model_name=nf_signal_1jet`.
(See [Output directory layout](../setup/usage.md#output-directory-layout)).
:::

## Chaining downstream tasks

The `requires` key creates ordered dependencies between downstream tasks:

```yaml
downstream_tasks:
  histogram:
    args: { ... }
  neyman:
    requires: ["histogram"]  # points to the previous entry
    args: { ... }
  eval:
    requires: ["neyman"]
    args: { ... }
```

When you run `DownstreamTask --downstream eval` (either backend), it runs `histogram` then
`neyman` then `eval`, checking output file existence to skip already-complete steps.
