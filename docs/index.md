# NEEDLE

**NEEDLE** is a workflow orchestrator for HEP machine learning pipelines, especially with
Neural Simulation Based Inference in mind. It combines
[LAW](https://law.readthedocs.io/en/latest/) or
[b2luigi](https://b2luigi.belle2.org/index.html) task scheduling,
[Lightning](https://lightning.ai/docs/pytorch/stable/) training modules, and
[Hydra](https://hydra.cc/docs/intro/) configuration management.

::: {admonition} One workflow to rule them all
:class: info
NEEDLE helps you set up a single scalable, reproducible and fully automated pipeline for training
all your neural networks in one go.
:::

---

::::{grid} 2
:gutter: 3

:::{grid-item-card} Setup and Introduction
:link: setup/index
:link-type: doc

Install the workspace and explore all the different ways to run your first Tasks
:::

:::{grid-item-card} Concepts
:link: concepts/task_hierarchy
:link-type: doc

Dive deeper into the mechanisms behind needle: the DAG Workflow, LAW and b2luigi backends and Hydra configs
:::

:::{grid-item-card} Examples
:link: examples/fair_universe_demo/index
:link-type: doc

**NEW** The FAIR Universe HiggsML demo with normalizing flows and classification.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Auto-generated reference for all public modules.
:::
::::

::: {admonition} Basic features
:class: note

With a minimal setup, NEEDLE gives you:

- Job submission to HTCondor or Slurm clusters, with no batch-system code to write
- [Automatic branching](concepts/hydra_config.md#the-expands-block) over estimators, systematic variations, ensembles and
    cross-validation folds for managing O(100)s of models
- A powerful [Hydra-based config composition](concepts/lightning_and_hydra_integration.md#hydra-from-the-cli)
    that allows flexible model exchange, experiment tracking and CLI overrides.
- [PyTorch Lightning training](concepts/lightning_and_hydra_integration.md#lightning) with check-pointing and MLflow logging built in
- [Easily access trained models](setup/usage.md#accessing-trained-models) using the `dag_snapshot.json` mapping every trained model to its
    checkpoint path, produced automatically at the end of a run
- Accessible from the Command Line Interface via `law run`/`b2luigi run`, or as a python package
:::

::: {admonition} Advanced features
:class: note

As your needs grow, NEEDLE also supports:

- Build complex model [inter-dependencies](concepts/hydra_config.md#the-requires-block)
- Two interchangeable backends, [LAW](concepts/law_tasks.md) and [b2luigi](concepts/b2luigi_tasks.md),
    sharing the same task definitions and config
- Extend your analysis post-training by importing needle Tasks or with needle's [DownstreamTasks](concepts/downstream_tasks.md),
    wired into the same dependency graph and with their own branch expansion.
- [Dask-Awkward](concepts/dask_awkward.md) data ingestion for parquet and ROOT files
- Training a single model directly, bypassing the full DAG, for fast iteration and debugging
:::

The starting point for using NEEDLE is the [Setup](setup/index.md) page, which shows how to install
the software.


## Libraries

The data-processing libraries are completely optional and are only used when selecting the builtin
NEEDLE modules in your config. For the training and inference, pytorch Lightning is a key
component that ensures all models are compatible with the framework. Finally, we use LAW or b2luigi
(both forks of Spotify's luigi) to schedule and organize Tasks — see
[DAG Workflow](concepts/task_hierarchy.md).

```{image} diagrams/website_technical_overview_light.png
:alt: libraries
:class: light-diagram
```

---

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

setup/index
setup/usage
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

concepts/task_hierarchy
concepts/law_tasks
concepts/b2luigi_tasks
concepts/lightning_and_hydra_integration
concepts/hydra_config
concepts/downstream_tasks
concepts/dask_awkward
```

```{toctree}
:maxdepth: 2
:caption: Examples
:hidden:

examples/fair_universe_demo/index
```

```{toctree}
:maxdepth: 2
:caption: NEEDLE API
:hidden:

api/index
```

```{toctree}
:maxdepth: 2
:caption: For Developers
:hidden:

for_devs/locally_build_docs
```
