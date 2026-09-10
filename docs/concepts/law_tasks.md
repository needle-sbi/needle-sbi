
# LAW Tasks

This page describes how `law` works in technical detail. For the overview on how to use it within
`needle-sbi`, see [Getting Started](../setup/index.md).

`law` is a fork of luigi that applies python task scheduling to HEP
environments. This page covers the `needle.tasks.law` backend, for the other backend, see
[b2luigi Tasks](b2luigi_tasks.md).

 - [luigi docs](https://luigi.readthedocs.io/en/stable/). Primary documentation
 - [law docs](https://law.readthedocs.io/en/latest/).
 - [columnflow docs](https://columnflow.readthedocs.io/en/latest/).
    Provides a bit of context to using law, even though most Tasks in ColumnFlow are fixed.

In addition, the `law` CLI tool provides tab-completion for available parameters.


## Running needle-sbi with backend law

`needle run` (the default backend is `law`) is the recommended entry point day-to-day. See
[Usage](../setup/usage.md). You can also skip it and drive the same tasks through the `law` CLI
directly, which additionally gives you tab-completion for all available arguments:

```bash
law run MainTask  # only training
law run DownstreamTask --downstream <name_from_config>  # training + post-training
```

`needle run <ClassName> --backend law ...` maps onto `law run <ClassName> ...` 1:1. The `--config-file` and `--results-path` are passed straight through. All other `--param key=value` becomes real`--key value` flag (dashes/underscores are interchangeable for `law`).

Some useful `law` args are listed here:

| Argument              | Description                                                  |
| --------------------- | ----------------------------------------------------- |
| `--local-scheduler`   | Use an in-memory central scheduler. Useful for testing. |
| `--help`              | Show most common flags and all task-specific flags. |
| `--log-file`          | A custom log file; default: `<task.default_log_file>`. |
| `--print-deps`        | Print task dependencies but do not run any task; this CSV parameter accepts a single integer value which sets the task recursion depth (0 means non-recursive). |
| `--print-status`      | Print the task status but do not run any task; this CSV parameter accepts up to three values: 1) the task recursion depth (0 means non-recursive), 2) the depth of the status text of target collections (default: 0), 3) a flag that is passed to the status text creation (default: `''`). |
| `--print-output`      | Print a flat list of output targets but do not run any task; this CSV parameter accepts up to two values: 1) the task recursion depth (0 means non-recursive), 2) a boolean flag that decides whether paths of file targets should contain file system schemes (default: `True`).       |
| `--remove-output`       | Remove task outputs but do not run any task by default; this CSV parameter accepts up to three values: 1) the task recursion depth (0 means non-recursive), 2) one of the modes `i` (interactive), `a` (all), `d` (dry run) (default: `i`), 3) a boolean flag that decides whether the task is run after outputs were removed (default: `False`). |

The extra `needle-sbi` arguments are

| Argument              | Description                                                  |
| --------------------- | -------------------------------------------------------------|
| `--config-file`       | Path to the Hydra config file (default: `conf/config.yaml`). |
| `--hydra-overrides`   | Overrides to be passed to hydra. Type `str`. Format: `'key1=value1 key2=value2'`, same as in the hydra docs (See [Hydra overrides](lightning_and_hydra_integration#step-3-runtime-overrides)). |
| `--results-path`      | Root directory where results are saved.                      |
| `--strict-config`     | Config conflict strictness: `IGNORE`, `WARN`, or `RAISE`. Whether same re-runs should also strictly have the same config. Default is `RAISE` |

## `law.Task` basics

This section also applies to `luigi` and `b2luigi`.

A Task is counted as complete if:

 1. **All its requirements are complete**. So all the Tasks that this Task depends on are marked as complete.
    This implies a recursive check that works up the DAG until it finds a Task that is not yet complete.
 2. **All its outputs exist**. Output files or folders are defined using the `output()` method. The outputs
    have to be created during the execution of the Task.

The `run()` method is responsible for actually executing the main body of code that the Task is supposed
to perform. If after reaching the end of the `run()` block an output file is missing, the Task is marked
as failed and the whole DAG stops. This is intended behavior since otherwise downstream tasks will
fail due to inexistent files that they in turn depend on.

A Task might require:

 1. Other Tasks using the `req()` method in law, or `requires()` in luigi. These are just other Tasks
    with associated parameters.
 2. Input files. In Law, `input()` provides a way to access the outputs of the required Tasks for this
    Task to use. Basically, you define `output()` with each output file having a fixed name and you
    access these names in the next Task using the `input()` method with that same name. If that file
    does not exist or the name is wrong, Law will raise an `Unfulfilled dependencies at RunTime` Error
    and tell you which files it expected.

## `TrainingTask` and batch submissions

The `TrainingTask` (`needle.tasks.law.training`) is the Task that runs an individual trainings. The
batch submission capability is added using the three `LocalWorkflow`, `HTCondorWorkflow` and
`SlurmWorkflow` mixins.

```python
class TrainingTask(
   BaseTrainingTask,
   LocalWorkflow,
   HTCondorWorkflow,
   SlurmWorkflow,
):
    ...
```

The `Workflow` mixins unlock the `--workflow=local|htcondor|slurm` CLI parameter. Adding
the `HTCondorWorkflow`/`SlurmWorkflow` mixins (`needle.tasks.law.workflows`) lets you run that
task on a batch system automatically. `DownstreamTask` (`needle.tasks.law.downstream`) uses the
same three mixins, so it supports batch dispatch too.

```bash
law run MainTask --TrainingTask-workflow htcondor
law run TrainingTask --workflow htcondor
```

Notice that `--TrainingTask-workflow` is a law pattern that indicates that TrainingTask should run
remotely, while MainTask, which is only a wrapper, remains local. You can replace `htcondor` with `slurm`
to change the batch system.


## The LAW config file (`law.cfg`)

`law.cfg` lives at the repo root and tells LAW where to find tasks and how to run them. The following
is an over-complete definition of the law cfg, with the defaults shown.

```ini
[modules]                           # where to find the Tasks
needle.tasks.law

[job]
job_file_dir: runs/law_outputs      # where to store logs
job_file_cleanup: False             # whether to delete logs after training
poll_interval: 30s                  # how often to check on jobs
retries: 2                          # how often to retry failed jobs

[logging]
luigi-interface: INFO               # logger level for luigi
law: INFO                           # logger level for law

[luigi_core]                        # luigi sections in law are marked luigi_<section_name>
workers: 4                          # how many workers to use
local_scheduler: True               # use a local scheduler, not a daemon
no_lock: True                       # allow parallel instances of the same Task

[luigi_worker]
keep_alive: True                    # re-use existing workers if possible
ping_interval: 20                   # how often to check on workers
wait_interval: 20                   # how long workers wait before asking for new jobs
max_reschedules: 0                  # whether to reschedule jobs

[luigi_scheduler]
retry_count: 0                      # whether to disable tasks that reached max retries

# These sections are only used as a fallback, for Tasks whose config-level `resources`
# dict is empty/unset (see "TrainingTask and batch submissions" above).
[luigi_TrainingTask_slurm]          # custom section for TrainingTask (SLURM)
nodes: 1                            # how many nodes to request
time: 60                            # how long to request these nodes for (unit depends on your system)
ntasks: 1                           # how many tasks per node
job-name: "needle-job"              # display name
partition: "cpuonly"                # which partition to use (here for DESY Maxwell)
mem-per-cpu: 2000                   # how much RAM to request (unit depends on your system)

[luigi_TrainingTask_htcondor]       # custom section for TrainingTask (HTCondor)
RequestRuntime: 600                 # how long to request for each job
RequestMemory: 2048                 # how much RAM to request (unit depends on your system)
RequestDisk: 10000000               # how much disk to request (unit depends on your system)
RequestCpus: 2                      # how many cores to request
```

The [luigi configuration](https://luigi.readthedocs.io/en/latest/configuration.html) page has all
the settings listed. The [law configuration](https://law.readthedocs.io/en/latest/config.html) explains
the `law` specific settings. The template created by `needle init` has the smallest usable config
already provided, you do not need to set all settings listed here. There is no section for local
resources, as these cannot be provisioned by `law`.
