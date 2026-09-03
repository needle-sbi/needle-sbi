# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
uv python pin 3.12
uv sync --group dev   # install runtime + dev dependencies
source .venv/bin/activate
source setup.sh       # sets LAW_HOME, LAW_CONFIG_FILE, SCRIPT_DIR, PYTHONPATH, shell completion
law index             # index LAW tasks in law.cfg (only needed for the law backend)
```

Key environment variables:
- `FAIR_UNIVERSE_DATA` — path to parquet file for dataset-dependent tests; empty string uses bundled test data
- `DELPHES_DATA_ROOT` / `DELPHES_DATA_PARQUET` — paths for Delphes-format test fixtures
- `LAW_HOME` / `LAW_CONFIG_FILE` — set by `setup.sh` (or `needle.api.configure_law()`); required for the LAW backend
- `SCRIPT_DIR` — project root, exported by `setup.sh`; read by the b2luigi backend (`needle.tasks.b2luigi.workflows.common.get_project_root`)

## Commands

**Tests:**
```bash
pytest                                                    # all non-slow, non-law, non-b2luigi tests
pytest -m slow                                            # slow tests, e.g. starting real law/b2luigi Tasks
pytest -m law                                             # law-backend-specific tests
pytest -m b2luigi                                         # b2luigi-backend-specific tests
pytest --benchmark-only                                   # benchmark tests, not used at this stage
```

Default markers exclude `slow`, `law`, and `b2luigi`; see `pyproject.toml` `[tool.pytest.ini_options]`.

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
needle init [directory]                     # scaffold a new NEEDLE project (law.cfg, settings.json, setup.sh, conf/)
needle init --backend law|b2luigi|both      # limit scaffolding to one backend (default: both)
needle init --no-conf                       # scaffold without the default conf/ directory
needle run MainTask --backend law           # shells out to `law run MainTask ...`
needle run MainTask --backend b2luigi       # runs in-process via b2luigi.process()
```

## Architecture

The needle is a **DAG workflow engine** layering three frameworks:
- **Luigi** — the shared task/dependency-graph model, wrapped by two interchangeable backends:
  **LAW** (remote job dispatch via HTCondor / Slurm, `needle.tasks.law`) and **b2luigi**
  (HTCondor / Slurm / LSF dispatch, in-process execution, `needle.tasks.b2luigi`)
- **Hydra** — structured configuration via dataclasses + YAML composition + CLI overrides
- **PyTorch Lightning** — training loop, checkpointing, MLflow logging inside the leaf task

See `docs/concepts/task_hierarchy.md` for the full rationale and a law-vs-b2luigi comparison table.

### Task DAG (`needle/tasks/`)

Task code is split into a backend-agnostic `base/` layer and two thin backend bindings,
`law/` and `b2luigi/`, that both subclass the same `base/` classes and expose identical task
names and parameters:

```
needle/tasks/
├── base/          # backend-agnostic luigi.Task logic (all the real behavior)
│   ├── expansion.py    # BaseExpansionTask: shared base for Estimator/Systematic/Ensemble/Fold
│   ├── main.py         # BaseMainTask
│   ├── training.py     # BaseTrainingTask (the Lightning training logic)
│   └── downstream.py   # BaseDownstreamMixin
├── law/           # law.Task subclasses binding base/ to law (LocalFileTarget, workflows/)
├── b2luigi/        # b2luigi.Task subclasses binding base/ to b2luigi (workflows/, settings.json)
└── mixins/hydra.py # HydraParamsMixin: config_file/hydra_overrides params, lazy MainConfig loading
```

Import `from needle.tasks.law import MainTask` or `from needle.tasks.b2luigi import MainTask` —
never mix backends within one DAG. `law` Tasks are law-only (not plain-luigi-compatible);
`b2luigi` Tasks remain fully compatible with plain `luigi.Task`.

Tasks form a strict hierarchy; each level `requires()` the level below it:

```
MainTask
 └── EstimatorTask          (one per estimator in config; fans out, writes a `.done` marker)
      └── SystematicTask    (one per systematic variation; `.done` marker)
           └── EnsembleTask (one per ensemble member; `.done` marker)
                └── FoldTask   (one per cross-validation fold; `.done` marker)
                     └── TrainingTask   ← actual Lightning training happens here (the only real leaf)
DownstreamTask              (wraps user luigi Tasks; supports branch expansion via `expands`)
```

- `EstimatorTask`/`SystematicTask`/`EnsembleTask`/`FoldTask` (`base/expansion.py`) are thin
  fan-out coordinators: their only output is a `.done` marker file, they do no training
  themselves.
- `TrainingTask` (`base/training.py`) is the sole leaf: it instantiates the Lightning `Trainer`,
  `LightningModule`, and `DataModule` via `hydra_instantiate`, runs `trainer.fit`, saves the
  checkpoint, and logs to MLflow. Pass `single=True` to train one estimator directly, ignoring
  `requires`/systematics/ensembles/folds entirely (see `needle.api.train_single`).
- `MainTask` (`base/main.py`) is the root entry point. It resolves and caches the full Hydra
  config to `<results_path>/config.yaml` before any subtasks run (a `strict_config` parameter,
  one of `IGNORE`/`WARN`/`RAISE`, controls how config conflicts with the cached version are
  handled), fans out to one `EstimatorTask` per estimator, and — once they complete — writes
  `<results_path>/dag_snapshot.json`: a flat `{"est=...&syst=...&ensem=...&fold=..." : ckpt_path}`
  dict of every trained checkpoint, built by walking the DAG (`snapshot_as_dict`).
- `DownstreamTask` wraps arbitrary user-defined `luigi.Task` subclasses declared under
  `downstream_tasks` in the config. It supports branch expansion (via `expands` in
  `DownstreamTaskConfig`) and can declare dependencies on other downstream tasks via `requires`.
  See `docs/concepts/downstream_tasks.md`.
- Workflow mixins in `needle/tasks/law/workflows/` (HTCondor, Slurm, local) and
  `needle/tasks/b2luigi/workflows/` (same, plus LSF, plus `configure_b2luigi()` for
  programmatic settings) provide execution backends.

### Configuration (`needle/utils/config_schema.py`)

Config is pure Python dataclasses (not Pydantic) registered in Hydra's ConfigStore. The hierarchy:

```
MainConfig
 ├── estimators: dict[str, EstimatorConfig]
 │    └── EstimatorConfig
 │         ├── expands: ExpansionConfig        ← controls task fan-out
 │         │    ├── systematics: dict[str, SystematicConfig]
 │         │    ├── ensembles: EnsembleConfig  (num_ensembles)
 │         │    └── folds: int
 │         └── requires: [str]                 ← inter-estimator deps
 ├── downstream_tasks: dict[str, DownstreamTaskConfig]
 ├── results_path: str
 ├── results_path_downstream: str
 └── custom_settings: Any
```

`needle/utils/config_utils.py` resolves and validates the full config (`initialize_hydra_config`,
`resolve_defaults`, `validate_graph` for cycle/missing-dependency detection, `compare_configs` for
cached-vs-new diffing) at startup. There is no separate results/aggregation config or module —
`dag_snapshot.json` is written directly by `MainTask` as a flat path dict; downstream consumers
resolve/aggregate checkpoints themselves.

### Public API (`needle/api/`)

High-level Python API for use outside of a running `law`/`b2luigi` process. Members are resolved
lazily via `needle/api/__init__.py:__getattr__` so `import needle.api` stays fast:

- `needle.api.config.Config` — load and resolve a Hydra config without LAW/b2luigi (wraps
  `initialize_hydra_config`; resolved `MainConfig` fields are accessible via attribute access or
  `Config.get("dot.notation.key")`)
- `needle.api.run.run()` / `RunResult` / `UnknownTaskError` — submit any task by class name to
  either backend (`backend="law"` shells out to `law run`; `backend="b2luigi"` runs in-process via
  `b2luigi.process()`)
- `needle.api.train.train_single()` — train one estimator directly via a `TrainingTask(single=True)`,
  bypassing the DAG fan-out entirely (b2luigi only; law has no in-process execution path)
- `needle.api.init.init()` / `InitResult` — scaffold a new project (used by `needle init`)
- `needle.api.law_settings.configure_law()` — set `LAW_HOME`/`LAW_CONFIG_FILE` from Python instead
  of sourcing `setup.sh`
- `needle.tasks.b2luigi.workflows.common.configure_b2luigi()` (re-exported as
  `needle.api.configure_b2luigi`) — set b2luigi batch-system settings from Python instead of
  `settings.json`

Note: `needle/api/__init__.py` also declares `Model`/`Dataset` entries in its lazy-attribute map
(`needle.api.model`/`needle.api.dataset`), but those modules do not currently exist in the
tree — accessing `needle.api.Model` or `needle.api.Dataset` raises `ModuleNotFoundError`. The
old `needle/evaluation/` package (pseudo-model / DAG-visualization code) has been removed entirely.

### ETL (`needle/etl/`)

Data ingestion layer built on Dask Awkward Arrays:

- `dask_ingestor.py` — `Ingestor`: lazy reader for parquet and ROOT files
- `array.py` — `NestedArrayIndexer` and helpers for awkward array manipulation
- `normalization.py` — feature normalisation utilities
- `conversion.py` — format conversion helpers

### ML (`needle/ml/`)

- `datasets/` — padded dataset implementations: `padded_eager.py` (in-memory), `padded_delayed_dask.py`,
  `padded_delayed_torch.py` (lazy/streamed), plus `io.py` and `kfold.py` helpers
- `lightning/datamodules/` — `padded_datamodule.py`, `pandas_datamodule.py` (Lightning `DataModule`s
  instantiated by `TrainingTask`)
- `lightning/models/` — `mock_transformer.py` (test/example model)

### TUI (`needle/tui/`)

Terminal UI components (`needle/tui/components/`), currently `version_info.py`.

### Workspace layout

```
needle-sbi/
├── containerization/    # Singularity/Apptainer container definitions
├── docs/                # Sphinx docs (MyST Markdown + RST API refs)
│   ├── concepts/        # task_hierarchy, law_tasks, b2luigi_tasks, downstream_tasks,
│   │                     # hydra_config, lightning_and_hydra_integration, dask_awkward
│   └── api/              # auto-generated RST reference, mirrors needle/ tree
│       ├── law_tasks/ b2luigi_tasks/   # per-backend task + mixins + workflows reference
│       └── needle_api/ needle_ml/ needle_utils/
├── examples/
│   └── fair_universe_demo/   # end-to-end demo (CNF estimators + classifier)
├── needle/              # Core library
│   ├── api/             # Public Python API (Config, run, train_single, init, configure_law/b2luigi)
│   ├── cli.py            # `needle` console-script entry point (init / run subcommands)
│   ├── etl/              # Dask/Awkward data ingestion
│   ├── ml/               # Lightning DataModules, datasets, models
│   │   ├── datasets/     # Padded dataset implementations (eager, dask, torch)
│   │   └── lightning/    # DataModules and mock model
│   ├── tasks/            # Task DAG: base/ (shared logic) + law/ + b2luigi/ backends + mixins/
│   ├── templates/        # Files scaffolded by `needle init` (law.cfg, settings.json, conf/, setup.sh)
│   ├── tui/               # Terminal UI components
│   └── utils/             # config_schema, config_utils, logging, dataclass, luigi_utils
├── tests/
│   ├── api/               # needle.api tests (Config, run, init, train_single)
│   ├── trainings/          # law/b2luigi task execution tests (marked `law`/`b2luigi`/`slow`)
│   ├── ml/ ingestion/ config/ benchmarks/
│   └── conf_tests/         # Hydra config used by tests (independent of example configs)
├── pyproject.toml
├── law.cfg               # LAW config (distinct from needle config.yaml); needed for law backend only
└── setup.sh
```

### Tests

- `tests/conf_tests/` — Hydra config used by all tests (independent of example configs)
- `conftest.py` provides: `make_parquet_file`, `ingestor`, `simple_sample` (parquet fixtures),
  `fair_universe_sample`, `fair_universe_demo_parquet`, `delphes_sample_root`, `delphes_sample_parquet`
  (env-gated fixtures that skip if the env var is unset), `config_factory()` (builds `MainConfig`
  with optional overrides), `config` (default config), and `dask_client` (session-scoped Dask
  `LocalCluster`)
- Tests exercising real law/b2luigi task execution are marked `law` / `b2luigi` (and usually `slow`)
  and excluded by default — see the `pytest -m law` / `pytest -m b2luigi` commands above
- `tests/trainings/` uses `tmp_path` to avoid collisions between concurrent runs
