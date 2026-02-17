# NEEDLE

**NEEDLE** is a workflow orchestrator for HEP machine learning pipelines, combining
[LAW](https://law.readthedocs.io/en/latest/) task scheduling,
[Lightning](https://lightning.ai/docs/pytorch/stable/) training modules, and
[Hydra](https://hydra.cc/docs/intro/) configuration management.

---

::::{grid} 2
:gutter: 3

:::{grid-item-card} Guides
:link: introduction
:link-type: doc

Narrative documentation covering the architecture, data handling with
dask-awkward, LAW task design, and Lightning/Hydra integration.
:::

:::{grid-item-card} API Reference
:link: api/orchestrator
:link-type: doc

Auto-generated reference for all public modules in the orchestrator,
law_tasks, ml, and preprocessor packages.
:::

::::

---

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

introduction
dask_awkward
law_tasks
lightning_and_hydra_integration
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

api/orchestrator
api/law_tasks
api/ml
api/preprocessor
```
