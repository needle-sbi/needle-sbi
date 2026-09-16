TrainingTask
============

The only real leaf of the law DAG: runs the PyTorch Lightning training loop for a single
fold. It is also the only law *workflow* task — multiple inheritance from
``LocalWorkflow``, ``HTCondorWorkflow``, and ``SlurmWorkflow`` gives it the
``--workflow=local|htcondor|slurm`` CLI switch. Every task above it (``FoldTask``,
``EnsembleTask``, …) is a plain ``.done``-marker that always runs locally.

Type Aliases
------------

.. autodata:: needle.tasks.law.training.TrainingTaskOutput

Reference
---------

.. automodule:: needle.tasks.law.training
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: law.Task
   :exclude-members: TrainingTaskOutput
