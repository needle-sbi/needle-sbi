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

:::{grid-item-card} Setup
:link: setup/index
:link-type: doc

Installation and running your first workflow.
:::

:::{grid-item-card} Concepts
:link: concepts/task_hierarchy
:link-type: doc

The DAG workflow and writing the Hydra config.
:::

:::{grid-item-card} Examples
:link: examples/fair_universe_demo/index
:link-type: doc

End-to-end example: FAIR Universe HiggsML demo with normalizing flows and classification.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Auto-generated reference for all public modules.
:::

::::

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
