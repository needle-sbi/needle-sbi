LAW Tasks
=========

Law Tasks are at the core of how NEEDLE organizes the training of your models. Since NSBI methods typically
rely on a combination of several neural networks, NEEDLE allows for a flexible graph building for
systematics, ensembling and cross-fold validation. Each layer of the Task graph gets its own Law Task.
The structure of the graph funnels from Folds down to a single MainTask.

.. image:: ../diagrams/dependency_graph.png
   :alt: LAW Tasks DAG Hierarchy
   :width: 90%
   :align: center

Core Tasks
----------

Main Task
~~~~~~~~~

Root entry point for the training DAG. Manages all estimators and coordinates training.

.. automodule:: law_tasks.main
   :members:

   :show-inheritance:

Estimator Task
~~~~~~~~~~~~~~

Trains a single estimator with all its systematic variations and ensemble runs.

.. automodule:: law_tasks.estimator
   :members:

   :show-inheritance:

Systematic Task
~~~~~~~~~~~~~~~

Handles a single systematic uncertainty variation with multiple ensemble runs.

.. automodule:: law_tasks.systematic
   :members:

   :show-inheritance:

Ensemble Task
~~~~~~~~~~~~~

Manages training for a single ensemble group across all cross-validation folds.

.. automodule:: law_tasks.ensemble
   :members:

   :show-inheritance:

Fold Task
~~~~~~~~~

Executes the actual training for a single cross-validation fold. Supports local,
HTCondor, and SLURM execution backends.

.. automodule:: law_tasks.fold
   :members:

   :show-inheritance:

Special Tasks
-------------

Snapshot Task
~~~~~~~~~~~~~

Collects all trained model checkpoints and creates a snapshot (dag_snapshot.json)
for evaluation and inference without re-running training.

.. automodule:: law_tasks.snapshot
   :members:

   :show-inheritance:

Downstream Task
~~~~~~~~~~~~~~~

Generic wrapper for post-training tasks. Enables flexible analysis pipelines
that run after the main training DAG completes. For more information, refer to the Concepts tab.

.. automodule:: law_tasks.downstream
   :members:

   :show-inheritance:

Mixins
------

Task Mixins provide reusable functionality for all tasks.

Hydra Mixin
~~~~~~~~~~~

Provides configuration management using Hydra. Automatically loads, resolves,
and caches configuration for all tasks.

.. automodule:: law_tasks.mixins.hydra
   :members:

   :show-inheritance:

Collect Output Mixin
~~~~~~~~~~~~~~~~~~~~

Interactive tool for debugging and exploring task output paths. Helps understand
task dependencies and outputs.

.. automodule:: law_tasks.mixins.collect_output
   :members:

   :show-inheritance:

Workflows
---------

Execution backends for different compute environments.

Common Workflow Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

Shared utilities for all workflow backends.

.. automodule:: law_tasks.workflows.common
   :members:

   :show-inheritance:

Local Workflow
~~~~~~~~~~~~~~

Execute tasks on local machine (single node or multiprocessing).

.. automodule:: law_tasks.workflows.local
   :members:

   :show-inheritance:

HTCondor Workflow
~~~~~~~~~~~~~~~~~

Execute tasks on HTCondor cluster (distributed scheduling).

.. automodule:: law_tasks.workflows.htcondor
   :members:

   :show-inheritance:

SLURM Workflow
~~~~~~~~~~~~~~

Execute tasks on SLURM cluster (HPC job scheduler).

.. automodule:: law_tasks.workflows.slurm
   :members:

   :show-inheritance:

Repository Bundling
~~~~~~~~~~~~~~~~~~~

Helper task for bundling repository for remote execution.

.. automodule:: law_tasks.workflows.bundle_repo
   :members:

   :show-inheritance:
