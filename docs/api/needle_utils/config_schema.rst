Config Schema
=============

.. autoclass:: needle.utils.config_schema.MainConfig
   :no-members:
   :no-undoc-members:

.. autoclass:: needle.utils.config_schema.EstimatorConfig
   :no-members:
   :no-undoc-members:

.. autoclass:: needle.utils.config_schema.SystematicConfig
   :no-members:
   :no-undoc-members:

.. autoclass:: needle.utils.config_schema.ExpansionConfig
   :no-members:
   :no-undoc-members:

.. autoclass:: needle.utils.config_schema.EnsembleConfig
   :no-members:
   :no-undoc-members:

.. note::
   ``AggregationConfig`` is currently **orphaned**: its only consumer, the fold/ensemble/
   systematic/estimator result-aggregation code in the old ``needle.utils.results`` and
   ``needle.tasks.base.snapshot`` modules, was removed. The dataclass still exists on
   ``MainConfig.aggregation`` but nothing in the current task DAG reads it.

.. autoclass:: needle.utils.config_schema.AggregationConfig
   :no-members:
   :no-undoc-members:

.. autoclass:: needle.utils.config_schema.DatasetConfig
   :no-members:
   :no-undoc-members:

.. autoclass:: needle.utils.config_schema.DownstreamTaskConfig
   :no-members:
   :no-undoc-members:
