# Writing Custom Downstream Tasks

Downstream tasks are how you attach your own analysis code to the NEEDLE pipeline. After all
models are trained and the snapshot is written, `DownstreamTask` instantiates and runs whatever
Luigi `Task` you specify in the config.

::: {admonition} When to use `DownstreamTask`
:class: info
`DownstreamTask` exists so you can point NEEDLE to your Task purely through config. If you are already have a `luigi`/`law`/`b2luigi` workflow or plan to build
one then importing `needle-sbi` as a package is simpler than to use `DownstreamTask`. In this case,
you only need to point your own `require()` method to NEEDLE's `MainTask` directly from your own
workflow, see the [Scenarios](#scenarios) section below.
:::

::: {question} Why is the luigi Task not in the DAG?
When using the DownstreamTask wrapper for your luigi Task, it is not possible for luigi or the other
schedulers to inspect your Task. This is because it would have to read the config file first in order
to see which DownstreamTasks are registered, but luigi only performs a static analysis. Therefore,
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

Whether a combination is known to work, based on what's actually exercised by the test suite:

|                       | plain `luigi`             | `law`                     | `b2luigi`             |
|-----------------------|---------------------------|---------------------------|-----------------------|
| **NEEDLE-in-Yours**   | Not supported (breaks batch submissions) | ☑ `needle.tasks.law`| ☑ `needle.tasks.b2luigi` |
| **Yours-in-NEEDLE** (`DownstreamTask`) |        ☑ |                         ☑ |                     ☑ |

`DownstreamTask` calls the wrapped task's `output()`/`run()`/`complete()` directly rather than
scheduling it, so the wrapped class only needs to look like a `luigi.Task` (duck typing, not an
`isinstance` check). Therefore, a plain `luigi.Task`, a `law.Task`, or a `b2luigi.Task` all work with
either backend's `DownstreamTask`.

::: {warning}
The `requires` method of your `luigi`/`law`/`b2luigi` Task is not used by `DownstreamTask`. You cannot
use this method to defined the dependency graph of your post-training. Instead, use the `requires`
section in the config.
:::

## Implementing a DownstreamTask

A downstream task is just a `luigi.Task`. NEEDLE does not impose any special base class.

```python
import luigi
import json


class MyAnalysisTask(luigi.Task):
    snapshot_path: str = luigi.Parameter()  # optional, see next Section
    output_path: "..."

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
  the task is marked failed and downstream tasks will not run.

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
| `args`                  | `dict[Any]`                       | Required. Needs at least the `_target_` field as an entry. All the other args are passed to your Task |
| `expands`               | `Optional[dict[str, list[Any]]]`  | How to duplicate this task. Use a descriptive name for each key and use a list of values to iterate over. If passing more than one key-value, then the cartesian product of those keys are used. See [expands block](./downstream_tasks.md#parameter-expansion) |

::: {note}
The `snapshot_path` parameter is injected automatically by `DownstreamTask`, you do not need
to specify it in the config. It is also completely optional, if your Task does not accept it, then
it will be dropped with an info message. The benefit is that you do not need to manually track the
location of the training output directory in your DownstreamTasks.
:::

::: {hint}
OmegaConf interpolations (`${...}`) are resolved before the task class is instantiated, so
`root_dir` will have the actual path string by the time `MyAnalysisTask.__init__` is called.
:::

## Running your DownstreamTasks

From `law`:

```bash
law run DownstreamTask \
    --downstream my_analysis \
    --config-file conf/config.yaml
```

From `needle` with law backend:

```bash
needle run DownstreamTask --backend law \
    --param downstream=my_analysis \
    --config-file conf/config.yaml
```

Or with the b2luigi backend:

```bash
needle run DownstreamTask --backend b2luigi \
    --param downstream=my_analysis \
    --config-file conf/config.yaml
```

Either way:
1. `MainTask` (and therefore the entire training pipeline) runs first if not already complete.
2. Any tasks listed in `requires` run if not already complete.
3. `MyAnalysisTask` is instantiated and run.

See [DAG Workflow](task_hierarchy.md) for how the two backends differ.

## Parameter expansion

If you want to run the same downstream task with different parameter values (e.g. validate each
model variant separately), use `expands`:

```yaml
downstream_tasks:
  validate_nf:  # mirroring the example/fair_universe_demo config
    args:
      _target_: my_package.tasks.ValidateNF
      root_dir: "${custom_settings.root_dir}"
    expands:
      model_name: ["nf_signal_1jet", "nf_background_1jet", "nf_signal_2jet"]
```

When `expands` lists multiple keys, NEEDLE spawns one branch per combination of the **cartesian
product** of all value lists (via `itertools.product`), not one branch per key. For example:

```yaml
expands:
  model_name: ["nf_signal_1jet", "nf_background_1jet"]
  jet_bin: ["1jet", "2jet"]
```

produces 2 × 2 = 4 branches (`(nf_signal_1jet, 1jet)`, `(nf_signal_1jet, 2jet)`,
`(nf_background_1jet, 1jet)`, `(nf_background_1jet, 2jet)`), each getting its own
`model_name`/`jet_bin` pair passed to the task constructor.

NEEDLE spawns one `DownstreamTask` per value in the expanded list. Each gets the extra parameter
passed to the task constructor:

```python
class ValidateNF(luigi.Task):
    model_name: str = luigi.Parameter()
    snapshot_path: str = luigi.Parameter()
    root_dir: str = luigi.Parameter()
    ...
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

When you run `DownstreamTask --downstream eval` (either backend), it runs `histogram` then
`neyman` then `eval`, checking output file existence to skip already-complete steps.
