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
- `LAW_HOME` / `LAW_CONFIG_FILE` — set by `setup.sh`; required for LAW task scheduling

## Commands

**Tests:**
```bash
pytest                                                    # all non-slow, non-benchmark tests
pytest -m slow                                            # slow tests, for example starting law Tasks
pytest --benchmark-only                                   # benchmark tests, not used at this stage
```

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

## Architecture

The orchestrator is a **DAG workflow engine** layering three frameworks:
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
SnapshotTask               (collects all checkpoints → dag_snapshot.json)
DownstreamTask             (generic post-training hook, waits on declared `requires`)
```

- `MainTask` is the root entry point. It resolves and caches the full Hydra config to `runs/config.yaml` before any subtasks run.
- `FoldTask` calls into `ml/` to instantiate the Lightning `Trainer`, `LightningModule`, and `DataModule`.
- `SnapshotTask` writes `dag_snapshot.json` mapping every (estimator, systematic, ensemble, fold) to its checkpoint path.
- `DownstreamTask` wraps arbitrary user-defined post-training tasks and can declare inter-estimator dependencies via the `requires` config field — this is the mechanism for multi-stage pipelines (e.g., train normalizing flows first, then use their outputs as input to a classifier).

### Configuration (`orchestrator/` + `conf/`)

Config is pure Pydantic dataclasses registered in Hydra's ConfigStore (`orchestrator/registry.py`). The hierarchy:

```
MainConfig
 └── EstimatorConfig[]
      ├── SystematicConfig[]
      ├── EnsembleConfig
      ├── expands: {systematics, ensembles, folds}   ← controls task fan-out
      └── requires: [str]                             ← inter-estimator deps
```

`orchestrator/config_utils.py` resolves and validates the full config (cycle detection, missing dependency checks, etc.) at startup. `orchestrator/results.py` defines result aggregation objects (`FoldResults`, `EnsembleResults`, …) that propagate up the DAG using configurable methods (`mean`, `weighted_mean`, `sum`).

### Workspace layout

This is a `uv` workspace with three members:
- `preprocessor/` — data ingestion from ROOT/Parquet, scaling, normalization
- `ml/` — Lightning modules, DataModules, datasets, network blocks
- `examples/fair_universe_demo/` — end-to-end demo (CNF signal estimators + classifier)

The root package (`orchestrator/`, `law_tasks/`, `conf/`) depends on both members.

### Tests

- `tests/hydra_test_conf/` — Hydra config used by all tests (independent of `conf/`)
- `conftest.py` provides `config_factory()` (builds `MainConfig` with optional overrides), `simple_sample` (parquet fixture), and a session-scoped Dask `LocalCluster` for benchmarks
- LAW tasks tests use `tmp_path` to avoid collisions between concurrent runs
