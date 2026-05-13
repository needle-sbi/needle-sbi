needle
======

A unified framework for flexible multi-model training and inference with automatic DAG-based orchestration.

Configuration & Utilities
-------------------------

Config Schema
~~~~~~~~~~~~~

.. automodule:: needle.utils.config_schema
   :members:
   :undoc-members:
   :show-inheritance:

Config Utils
~~~~~~~~~~~~

.. automodule:: needle.utils.config_utils
   :members:
   :undoc-members:
   :show-inheritance:

Dataclass Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.utils.dataclass
   :members:
   :undoc-members:
   :show-inheritance:

Array & File Utilities
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.utils.array
   :members:
   :undoc-members:
   :show-inheritance:

Data Conversion
~~~~~~~~~~~~~~~

.. automodule:: needle.utils.conversion
   :members:
   :undoc-members:
   :show-inheritance:

Dask Data Ingestor
~~~~~~~~~~~~~~~~~~

.. automodule:: needle.utils.dask_ingestor
   :members:
   :undoc-members:
   :show-inheritance:

Results & Snapshots
~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.utils.results
   :members:
   :undoc-members:
   :show-inheritance:

Logging
~~~~~~~

.. automodule:: needle.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:

Timing & Performance
~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.utils.epoch_timer
   :members:
   :undoc-members:
   :show-inheritance:

Normalization
~~~~~~~~~~~~~

.. automodule:: needle.utils.normalization
   :members:
   :undoc-members:
   :show-inheritance:

LAW/Luigi Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.utils.luigi_utils
   :members:
   :undoc-members:
   :show-inheritance:

ML: Datasets
-----------

Dataset Base Classes
~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.ml.datasets.padded_base
   :members:
   :undoc-members:
   :show-inheritance:

Dataset I/O
~~~~~~~~~~~

.. automodule:: needle.ml.datasets.io
   :members:
   :undoc-members:
   :show-inheritance:

K-Fold Dataset
~~~~~~~~~~~~~~

.. automodule:: needle.ml.datasets.kfold
   :members:
   :undoc-members:
   :show-inheritance:

Eager Padded Dataset
~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.ml.datasets.padded_eager
   :members:
   :undoc-members:
   :show-inheritance:

Delayed Torch Dataset
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.ml.datasets.padded_delayed_torch
   :members:
   :undoc-members:
   :show-inheritance:

Delayed Dask Dataset
~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.ml.datasets.padded_delayed_dask
   :members:
   :undoc-members:
   :show-inheritance:

ML: Lightning
-------------

Padded DataModule
~~~~~~~~~~~~~~~~~

.. automodule:: needle.ml.lightning.datamodules.padded_datamodule
   :members:
   :undoc-members:
   :show-inheritance:

Mock Transformer Model
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.ml.lightning.models.mock_transformer
   :members:
   :undoc-members:
   :show-inheritance:

API: High-Level Interface
-------------------------

Configuration Manager
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.api.config
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Loader
~~~~~~~~~~~~~~

.. automodule:: needle.api.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Model Wrapper
~~~~~~~~~~~~~

.. automodule:: needle.api.model
   :members:
   :undoc-members:
   :show-inheritance:

Training API
~~~~~~~~~~~~

.. automodule:: needle.api.train
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation & Pseudo-Models
---------------------------

Pseudo Model
~~~~~~~~~~~~

.. automodule:: needle.evaluation.pseudo_model
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Pseudo Model
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.evaluation.pseudo_model_parallel
   :members:
   :undoc-members:
   :show-inheritance:

Vectorized Pseudo Model
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.evaluation.pseudo_model_vectorized
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark Profiling
~~~~~~~~~~~~~~~~~~~

.. automodule:: needle.evaluation.benchmark_detailed
   :members:
   :undoc-members:
   :show-inheritance:

DAG Visualization
~~~~~~~~~~~~~~~~~

.. automodule:: needle.evaluation.dag_visualization
   :members:
   :undoc-members:
   :show-inheritance:
