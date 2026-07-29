HydraMixin
==========

.. note::
   This module is a backwards-compatible alias. The real implementation is the
   backend-agnostic ``needle.tasks.mixins.hydra.HydraParamsMixin``, shared by both the law
   and b2luigi backends. ``law.Parameter`` re-exports ``luigi.Parameter`` unchanged, so no
   law-specific logic is needed here.

Provides configuration management using Hydra. Automatically loads, resolves,
and caches configuration for all tasks.

.. automodule:: needle.tasks.law.mixins.hydra
   :members:
   :undoc-members:
   :show-inheritance:
