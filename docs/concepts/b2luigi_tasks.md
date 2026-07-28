# b2luigi Tasks

[B2luigi](https://b2luigi.belle2.org/index.html) is a fork of luigi by the Belle II
collaboration. Compared to LAW, it is notably simpler and has very nice documentation. This page covers the
`needle.tasks.b2luigi` backend specifically — for the backend-agnostic DAG explanation see
[DAG Workflow](task_hierarchy.md). For the other backend, see [LAW Tasks](law_tasks.md).

 - [luigi docs](https://luigi.readthedocs.io/en/stable/)
 - [B2luigi docs](https://b2luigi.belle2.org/index.html)

The only real `b2luigi.Task` is the actual TrainingTask, which requires `b2luigi` for the batch
submission, the other Tasks are thin wrappers around regular `luigi`. Understanding `luigi` already
primes you to understand the added features of `b2luigi` intuitively.

## Simple example

```python
from typing import Type

import b2luigi
import luigi

from needle.tasks.base.fold import BaseFoldTask


class FoldTask(BaseFoldTask, b2luigi.Task):
    task_namespace = "b2luigi"

    def _training_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.training import TrainingTask
        return TrainingTask  # <- just point to the b2luigi implementation
```

::: {admonition} "Why `task_namespace = "b2luigi"`?"
:class: info
Luigi's global task registry keys tasks by `family = f"{namespace}.{classname}"` when a task is
namespaced. Without it, `b2luigi.FoldTask`/`EnsembleTask`/etc. would collide in the registry
with the identically-named classes registered by `needle.tasks.law`. Since both
backends may be imported in the same process (e.g. by `needle/cli.py` or in tests), the
namespace is what separates them.
:::

## Batch submissions

When running from `needle run` CLI, the settings are listed in the `settings.json` file at the project root:

```json
{
  "batch_system": "htcondor",
  "htcondor_settings": {
    "request_memory": "2048MB",
    "request_cpus": 2,
    "+RequestRuntime": 600
  }
}
```

For experts directly using `needle` Tasks in their own workflows, the settings can also be
accessed from pure python using the `configure_b2luigi` function:

```python
from needle.tasks.b2luigi.workflows.common import configure_b2luigi

configure_b2luigi(results_path="runs", batch_system="htcondor")
```

Or per-task, by overriding `htcondor_settings`/`slurm_settings` class attributes in the given
Task.

## Running needle-sbi with backend b2luigi

Running the Tasks from the CLI:

```bash
needle run MainTask --backend b2luigi --batch-system htcondor --workers 4
needle run DownstreamTask --backend b2luigi --param downstream=<name_from_config>
```

`needle run --backend b2luigi` looks up the task class by name. It temporarily clears `sys.argv` around
that call, since `b2luigi.process()` parses `sys.argv` itself for its own flags (`--batch`, `--test`, ...)
and would otherwise conflict with needle's own argument parser.
With this approach, you need to specify `--param downstream` instead of the direct `--downstream`.
