DownstreamTask
==============

Generic wrapper for post-training tasks. Enables flexible analysis pipelines that run after
the main training DAG completes. Reimplements the law backend's ``expands`` branch expansion
by hand — via a URL-encoded ``branch_params`` parameter and a "root combinator" task instance
— since there is no ``law.LocalWorkflow`` equivalent in plain b2luigi/luigi. For more
information, refer to the Concepts tab.

.. automodule:: needle.tasks.b2luigi.downstream
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: b2luigi.Task
