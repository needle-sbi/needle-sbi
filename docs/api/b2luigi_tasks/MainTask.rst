MainTask
========

Root entry point for the training DAG. Fans out to one ``EstimatorTask`` per estimator in
the config, then writes ``dag_snapshot.json`` with all trained checkpoint paths.

.. automodule:: needle.tasks.b2luigi.main
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: b2luigi.Task
