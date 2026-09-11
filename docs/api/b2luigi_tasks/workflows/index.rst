Workflows
=========

Unlike the law backend, b2luigi has no per-batch-system Python modules (no
``HTCondorWorkflow``/``SlurmWorkflow`` mixin classes) — batch dispatch is just settings keys
consumed internally by b2luigi itself, configured via :doc:`common`.

.. toctree::
   :maxdepth: 2

   common
