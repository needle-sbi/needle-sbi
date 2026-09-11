HydraMixin
==========

.. note::
   This module is a naming-compatibility alias (matching the ``law_tasks``/``mixins``
   convention). The real implementation is the backend-agnostic
   ``needle.tasks.mixins.hydra.HydraParamsMixin``, shared with the law backend. There is no
   b2luigi analogue of law's ``CollectOutputMixin`` — no equivalent interactive
   output-collection tooling exists for this backend.

Provides configuration management using Hydra. Automatically loads, resolves,
and caches configuration for all tasks.

.. automodule:: needle.tasks.b2luigi.mixins
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
