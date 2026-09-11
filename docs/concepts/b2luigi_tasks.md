# b2luigi Tasks

[B2luigi](https://b2luigi.belle2.org/index.html) is a fork of luigi by the Belle II
collaboration. Compared to LAW, it is notably simpler and has very nice documentation. This page covers the
`needle.tasks.b2luigi` backend specifically — for the backend-agnostic DAG explanation see
[DAG Workflow](task_hierarchy.md). For the other backend, see [LAW Tasks](law_tasks.md).

 - [luigi docs](https://luigi.readthedocs.io/en/stable/)
 - [B2luigi docs](https://b2luigi.belle2.org/index.html)

The real `b2luigi.Task`s are `TrainingTask` and `DownstreamTask`, both of which support batch
submission. The other Tasks (`MainTask`, `EstimatorTask`, `SystematicTask`, `EnsembleTask`, `FoldTask`) all run locally and are very lightweight.
Understanding `luigi`
already primes you to understand the added features of `b2luigi` intuitively.


## Batch submissions

When running from the `b2luigi run` CLI, the settings are listed in the `settings.json` file at the project root:

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
import needle

needle.configure_b2luigi(
  batch_system="htcondor",
  env_script=None,
)
```

::: {warning}
By default, NEEDLE sets `htcondor_settings = {"getenv": "True"}` in order to directly ship the
environment used by the submitter to the worker node. If you wish to avoid this, you must point
`env_script` to a custom setup script that sources your environment. By default, `env_script` will
resolve to `setup.sh`.
See [Troubleshooting](../setup/usage.md#troubleshooting)
:::

## Running from `b2luigi run`

Get tab-completion with:

```bash
b2luigi --install-completion
```

Run Tasks with:

```bash
b2luigi run MainTask --batch
b2luigi run DownstreamTask --param downstream=<name_from_config>
```

::: {important}
Batch submission (`--batch`) re-invokes the task on the worker node via the real
`b2luigi batch-runner` CLI command, which unconditionally imports a `tasks.py` file at
the project root to resolve the task class. `needle init --backend b2luigi` scaffolds this
`tasks.py` for you.
Make sure it exists and that the `b2luigi` console script is on `PATH` on worker nodes too.
Purely local runs (no `--batch`) do not need `tasks.py`.
:::

From Python (e.g. a notebook), `needle.api.run()` is the equivalent entry point: it calls the
same `b2luigi.cli.utils.process_task_instance()` the `b2luigi run` CLI command itself uses.

### Removing task outputs

The `b2luigi` CLI ships a dedicated `remove` subcommand, equivalent to `law`'s `--remove-output`:

```bash
b2luigi remove MainTask --with-requirements
b2luigi remove DownstreamTask --param downstream=<name_from_config> --with-requirements
```

Without `--with-requirements`, only the named task's own outputs are removed, not its dependencies'.
Pass `-y`/`--yes` to skip the confirmation prompt (e.g. in scripts), and `--keep <ClassName,...>` to
protect specific task classes from removal.
