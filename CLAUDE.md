# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
uv python pin 3.12
uv sync --group dev   # install runtime + dev dependencies
source .venv/bin/activate
source setup.sh       # sets LAW_HOME, LAW_CONFIG_FILE, PYTHONPATH, shell completion
law index             # index LAW tasks in law.cfg
```

Key environment variables:
- `FAIR_UNIVERSE_DATA` — path to parquet file for dataset-dependent tests; empty string uses bundled test data
- `DELPHES_DATA_ROOT` / `DELPHES_DATA_PARQUET` — paths for Delphes-format test fixtures
- `LAW_HOME` / `LAW_CONFIG_FILE` — set by `setup.sh`; required for LAW task scheduling

## Commands

**Tests:**
```bash
pytest                                                    # all non-slow, non-law, non-benchmark tests
pytest -m slow                                            # slow tests, for example starting law Tasks
pytest --benchmark-only                                   # benchmark tests, not used at this stage
```

Default markers exclude `slow` and `law`; see `pyproject.toml` `[tool.pytest.ini_options]`.

**Lint / format:**
```bash
black .
isort .
flake8 .
mypy .
pre-commit run --all-files
```

Line length is 120. mypy uses `disallow_untyped_defs = true`.

**Docs:**
```bash
uv sync --group docs
uv run python -m sphinx -T -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

**CLI:**
```bash
needle init [directory]   # scaffold a new NEEDLE project (law.cfg, setup.sh, conf/)
needle init --no-conf     # scaffold without the default conf/ directory
```

## Architecture

The needle is a **DAG workflow engine** layering three frameworks:
- **LAW (Luigi)** — task scheduling, dependency tracking, checkpointing, remote job dispatch (HTCondor / Slurm)
- **Hydra** — structured configuration via dataclasses + YAML composition + CLI overrides
- **PyTorch Lightning** — training loop, checkpointing, logging inside each leaf task

### Task DAG (`law_tasks/`)

Tasks form a strict hierarchy; each level `requires()` the level below it:

```
MainTask
 └── EstimatorTask          (one per estimator in config)
      └── SystematicTask    (one per systematic variation)
           └── EnsembleTask (one per ensemble group)
                └── FoldTask  ← actual Lightning training happens here
SnapshotTask               (requires MainTask → writes dag_snapshot.json with nodes + edges)
DownstreamTask             (law.LocalWorkflow; wraps user luigi Tasks; supports branch expansion)
```

- `MainTask` is the root entry point. It resolves and caches the full Hydra config to `<results_path>/config.yaml` before any subtasks run. A `ConfigStrictness` enum (IGNORE / WARN / RAISE) controls how config conflicts with the cached version are handled.
- `FoldTask` calls into `needle/ml/` to instantiate the Lightning `Trainer`, `LightningModule`, and `DataModule`.
- `SnapshotTask` writes `dag_snapshot.json` as a `DAGSnapshot` object containing typed `ModelNodeMetadata` nodes and `AggregationEdge` edges for every (estimator, systematic, ensemble, fold) trained.
- `DownstreamTask` is a `law.LocalWorkflow` that wraps arbitrary user-defined `luigi.Task` subclasses. It supports branch expansion (via `expands` in `DownstreamTaskConfig`) and can declare dependencies on other downstream tasks via `requires`.
- Workflow mixins in `law_tasks/workflows/` provide HTCondor, Slurm, and local execution backends for `FoldTask`.

### Configuration (`needle/utils/config_schema.py`)

Config is pure Python dataclasses (not Pydantic) registered in Hydra's ConfigStore. The hierarchy:

```
MainConfig
 ├── estimators: dict[str, EstimatorConfig]
 │    └── EstimatorConfig
 │         ├── expands: ExpansionConfig        ← controls task fan-out
 │         │    ├── systematics: dict[str, SystematicConfig]
 │         │    ├── ensembles: EnsembleConfig
 │         │    └── folds: int
 │         └── requires: [str]                 ← inter-estimator deps
 ├── downstream_tasks: dict[str, DownstreamTaskConfig]
 ├── aggregation: AggregationConfig            ← fold/ensemble/systematic/estimator aggregation methods
 ├── results_path: str
 ├── results_path_downstream: str
 └── custom_settings: Any
```

`needle/utils/config_utils.py` resolves and validates the full config (cycle detection, missing dependency checks, defaults resolution) at startup. `needle/utils/results.py` defines result objects (`FoldResults`, `EnsembleResults`, …) and the `DAGSnapshot` / `ModelNodeMetadata` / `AggregationEdge` types used by `SnapshotTask` and the pseudo-models.

### Public API (`needle/api/`)

High-level Python API for use outside of LAW tasks:

- `needle.api.config.Config` / `config()` — load and resolve a Hydra config without LAW
- `needle.api.model.Model` / `model()` — load a trained `DAGSnapshot` as a callable `PseudoModel`
- `needle.api.dataset.Dataset` / `dataset()` — instantiate a Lightning DataModule from a resolved config
- `needle.api.train.train_single_lightning_module()` — train a single Lightning module directly

### Evaluation (`needle/evaluation/`)

Three implementations of the ensemble pseudo-model that loads from `dag_snapshot.json`:

- `pseudo_model.py` — `PseudoModel`: sequential aggregation
- `pseudo_model_parallel.py` — `NEEDLEParallel`: parallel evaluation
- `pseudo_model_vectorized.py` — `NEEDLEVectorized`: vectorized batched evaluation
- `dag_visualization.py` — DAG visualisation utilities

### ETL (`needle/etl/`)

Data ingestion layer built on Dask Awkward Arrays:

- `dask_ingestor.py` — `Ingestor`: lazy reader for parquet and ROOT files
- `array.py` — `NestedArrayIndexer` and helpers for awkward array manipulation
- `normalization.py` — feature normalisation utilities
- `conversion.py` — format conversion helpers

### Workspace layout

```
needle-sbi/
├── containerization/    # Singularity/Apptainer container definitions
├── docs/                # Sphinx docs (MyST Markdown + RST API refs)
├── examples/
│   └── fair_universe_demo/   # end-to-end demo (CNF estimators + classifier)
├── law_tasks/           # LAW workflow task classes
│   ├── mixins/          # HydraMixin, CollectOutputMixin
│   └── workflows/       # HTCondor, Slurm, local execution backends
├── needle/              # Core library
│   ├── api/             # Public Python API (Config, Model, Dataset, train)
│   ├── etl/             # Dask/Awkward data ingestion
│   ├── evaluation/      # PseudoModel variants, DAG visualisation
│   ├── ml/              # Lightning DataModules, datasets, models
│   │   ├── datasets/    # Padded dataset implementations (eager, dask, torch)
│   │   └── lightning/   # DataModule and mock model
│   ├── templates/       # Files scaffolded by `needle init`
│   └── utils/           # config_schema, config_utils, results, logging, …
├── tests/
│   ├── conf_tests/      # Hydra config used by tests (independent of examples/conf/)
│   └── …
├── tui/                 # Terminal UI components
├── pyproject.toml
├── law.cfg              # LAW config (distinct from needle config.yaml)
└── setup.sh
```

### Tests

- `tests/conf_tests/` — Hydra config used by all tests (independent of example configs)
- `conftest.py` provides: `make_parquet_file`, `ingestor`, `simple_sample` (parquet fixtures), `fair_universe_sample`, `delphes_sample_root`, `delphes_sample_parquet` (env-gated fixtures that skip if env var unset), `config_factory()` (builds `MainConfig` with optional overrides), `config` (default config), and `dask_client` (session-scoped Dask `LocalCluster`)
- LAW tasks tests use `tmp_path` to avoid collisions between concurrent runs
