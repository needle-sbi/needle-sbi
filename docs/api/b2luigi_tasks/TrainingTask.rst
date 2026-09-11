TrainingTask
============

The only real leaf of the b2luigi DAG, and the only class in this backend that inherits
``b2luigi.Task`` — runs the PyTorch Lightning training loop for a single fold. Unlike the law
backend, there is no workflow-mixin/branch-map machinery: batch dispatch is configured
globally via :func:`~needle.tasks.b2luigi.workflows.common.configure_b2luigi` (or a
``settings.json`` file), or per-task via ``htcondor_settings`` / ``slurm_settings`` class
attributes.

.. automodule:: needle.tasks.b2luigi.training
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: b2luigi.Task
