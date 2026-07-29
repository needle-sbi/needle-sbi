# DAG Workflow

## Why use a DAG for NSBI training?

Many NSBI tools rely on large neural surrogates to estimate statistical quantities. The final estimator
is usually composed of several sub-models, in some cases up to hundreds of networks. In `needle-sbi`, the
training of the models is individualized into separate python Tasks, with each Task being
a node in the larger Task graph. The edges of the graph represent a dependency: Task B depends
on Task A meaning A must finish before B can start.

::: {admonition} Advantages of DAG workflows
:class: tip
- It makes the dependency structure explicit and reproducible, no loose scripts.
- You can run the **whole pipeline** from a single command
- Networks that depend on each other are trained in the proper order
- Partially complete runs can be resumed: the orchestrator re-checks which outputs exist and
  only re-runs what is missing.
- The same task graph can be executed locally, on SLURM, or on HTCondor without changing task code,
  only user settings.
:::

## Python based workflows: Luigi, b2luigi and law

The `needle-sbi` package ships this DAG in three forms, all in python, depending on your needs:

 1. The base `luigi` layer that implements the structure of the graph and the computing logic.
 2. A `b2luigi` wrapper (batch submissions), see [b2luigi Tasks](b2luigi_tasks.md)
 3. A `law` wrapper (batch submissions), see [LAW Tasks](law_tasks.md)

```{image} ../diagrams/luigi_inheritance_chart.png
:alt: luigi task inheritance chart
:class: light-diagram
:width: 70%
:align: center
```

The two wrappers are virtually identical, same graph, same parameters and execution logic. The only
difference is the way how they dispatch trainings to HPCs. Users can choose either `b2luigi` or `law`
based on their own preference and existing knowledge. In both cases, you can easily expand the DAG
tree by importing the `needle-sbi` `b2luigi`/`law` Tasks in your own workflow. Note that while `b2luigi`
Tasks are fully compatible with regular `luigi` Tasks, `law` Tasks are not, so in the latter case
you must work within `law`. We still provide a way to append regular `luigi` Tasks to your `law`
workflow using [DownstreamTasks](downstream_tasks.md).

:::{admonition} Should I use b2luigi or law?
:class: hint

This comes down to personal preference.
 - `law` has been long established within the CMS experiment
and has expanded towards other experiments at the LHC. It has many features but lacks a good
documentation.
 - `b2luigi` on the other hand is slightly newer and is mainly used by the Belle II
collaboration. It has excellent documentation and intuitive usage.
:::

(choosing-a-backend)=
## Choosing a backend

|                                       | LAW                                       | b2luigi                       |
|---------------------------------------|-------------------------------------------|-------------------------------|
| Compatible with plain luigi           |                                           | ✅                            |
| local                                 | ✅                                        | ✅                            |
| SLURM (batch)                         | ✅                                        | ✅                            |
| HTCondor (batch)                      | ✅                                        | ✅                            |
| LSF (batch)                           |                                           | ✅                            |
| Settings file                         | `law.cfg`                                 | `settings.json`               |
| Running natively (CLI)                | `law run MainTask ...`                    |                               |
| Running from `needle-sbi` (CLI)       | `needle run MainTask ...`                 | `needle run --backend b2luigi MainTask`      |
| Importing `needle-sbi` Tasks (python) | `from needle.tasks.law import MainTask `  | `from needle.tasks.b2luigi import MainTask`  |

Both backends implement the exact same DAG shape described above — pick whichever fits your
batch system and existing tooling. See [LAW Tasks](law_tasks.md) and
[b2luigi Tasks](b2luigi_tasks.md) for the concrete APIs.

## The NEEDLE Training DAG

```{image} ../diagrams/dependency_graph_full.png
:alt: dependency graph
:class: light-diagram
:width: 70%
:align: center
```

This structure is over-complete on purpose: if you only have one type of model, or no systematics to
train on, or do not need ensembling or cross-fold validation, then the graph skips these layers.

You only need to know about `MainTask` (only training) and `DownstreamTask` (training and post-training),
the other Tasks are intermediaries that organize the execution.

We are working on also allowing `UpstreamTasks` which would run *before* the whole training pipeline
using the same mechanism as `DownstreamTasks`.

## Extending the DAG

You can append your own tasks to the DAG, for example validation steps or even performing the
fits using the trained weights. This is possible in the following ways:

 1. Register your luigi Tasks in needle with [DownstreamTasks](downstream_tasks) to append regular
    `luigi` Tasks to the workflow. Works in both `law` and `b2luigi` backends. Here the workflow is
    still controlled by `needle-sbi`.
 2. In `law`, import our Tasks into your workflow from `needle.tasks.law` (`MainTask` for example):

      ```python
      from needle.tasks.law import MainTask
      import law


      class SimpleTask(law.Task):
          def requires():
              return MainTask()
      ```

    Which you can run from the CLI using `law run <my_script>.SimpleTask`.

 3. In `b2luigi`, same but import from `needle.tasks.b2luigi` (same `MainTask` name):

      ```python
      from needle.tasks.b2luigi import MainTask
      import b2luigi


      class SimpleTask(b2luigi.Task):
          def requires(self):
              return MainTask()


      if __name__ == "__main__":
          b2luigi.process(SimpleTask())
      ```

      Which allows you to run it from the CLI with `python3 <my_script>.py`.
