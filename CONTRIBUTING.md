# Contributing to needle-sbi

This document describes how to propose changes to this repository — for human contributors and
for AI coding agents (e.g. Claude Code) working in this repo. It complements `CLAUDE.md`, which
covers environment setup and architecture; this file covers *workflow*: how work is branched,
named, tested, and merged.

## Quick checklist

Before opening a PR:

- [ ] Branch follows the naming convention below
- [ ] `pre-commit run --all-files` passes (`black`, `isort`, the `pre-commit-hooks` set)
- [ ] `flake8 .` and `mypy .` pass (run manually — not part of the default `pre-commit` stage, see below)
- [ ] `pytest` passes locally (add `-m slow`, `-m law`, `-m b2luigi` runs if you touched those areas)
- [ ] New/changed behavior has test coverage under `tests/`
- [ ] Docs updated if you changed public API, config schema, or task DAG behavior (`docs/concepts/`,
      docstrings)
- [ ] Commit messages and PR description explain *why*, not just *what*
- [ ] PR description links the issue it resolves (`Closes #123`) if one exists

## Branching model

- `main` is always releasable. Do not push directly to `main`; all changes land via PR.
- Long-running integration branches (e.g. `dev_*`) may exist for large multi-PR efforts, but new
  work should still branch off `main` and target `main` unless explicitly coordinating a staged
  migration — say so in the PR description if you target something other than `main`.
- Every change gets its own branch. Do not stack unrelated changes on one branch.

### Branch naming

```
<type>/<short-slug>              # e.g. feat/b2luigi-lsf-backend
<type>/issue-<n>-<short-slug>    # e.g. fix/issue-42-xrootd-logging
```

`<type>` is one of:

| type       | use for                                                              |
|------------|----------------------------------------------------------------------|
| `feat`     | new functionality                                                    |
| `fix`      | bug fix                                                              |
| `refactor` | internal restructuring with no behavior change                       |
| `docs`     | documentation only                                                   |
| `test`     | tests only (no production code change)                               |
| `chore`    | tooling, CI, dependency bumps                                        |
| `perf`     | performance work (e.g. ingestion/profiling changes)                  |

`<short-slug>` is lowercase, hyphen-separated, 2-5 words — enough to identify the change without
opening the branch. Include the issue number when the branch resolves a tracked issue.

If you are an agent working from a worktree, the worktree/branch name your tool generates
(`worktree-...`) should try to match this convention. Renaming or noting the intended
branch name in the PR title/description so history stays legible is also preferred.

## Commit messages

- Subject line: imperative mood, no trailing period, ideally under 72 chars
  (e.g. `Fix XRootD logging duplication`, not `Fixed` or `Fixes`).
- Avoid `[WIP]` commits on branches that are about to be merged — squash or amend before opening
  the PR, or mark the PR itself as a GitHub draft instead.
- Avoid noise merge commits (`Merge branch 'main' into ...`) accumulating on a feature branch;
  prefer rebasing on `main` before opening/updating a PR. If a merge commit is unavoidable, that's
  fine — just don't let a PR branch collect a long chain of them.
- Reference the issue in the body when relevant (`Refs #42`, `Closes #42`).

## Pull requests

- **Title**: same conventions as commit subjects — imperative, specific
  (e.g. `Add LSF workflow mixin to b2luigi backend`, not `Updates`).
- **Description** should cover:
  - What changed and why (link the issue with `Closes #N` / `Refs #N` if one exists)
  - Which backend(s) are affected (`law`, `b2luigi`, both, or neither)
  - Test plan: what you ran (`pytest`, `pytest -m law`, `pytest -m b2luigi`, manual `law run` /
    `b2luigi run` check) and what env vars were needed (`FAIR_UNIVERSE_DATA`, `DELPHES_DATA_ROOT`, ...)
  - Any doc updates made or needed
- Keep PRs scoped to one logical change. If a PR grows to touch unrelated areas
  (e.g. a bug fix plus a refactor plus new config), split it.
- Draft PRs are fine and encouraged for `[WIP]` work-in-progress instead of `[WIP]` commit prefixes.
- Squash-merge is preferred for PRs with noisy/WIP commit history; a clean sequence of atomic
  commits can be merged as-is.
- At least one review approval is required before merging, except for trivial doc-only or CI-only
  changes at the maintainers' discretion.

## Issues

When filing an issue, include:

- **What backend(s)** are involved (`law`, `b2luigi`, both, neither/general)
- **Repro steps** or the config/command that triggers the problem
- **Expected vs. actual behavior**
- **Environment**: Python version, whether `FAIR_UNIVERSE_DATA`/`DELPHES_DATA_ROOT` were set, LAW
  vs b2luigi, relevant package versions if not obviously the tip of `main`

Feature requests should state the use case, not just the desired API — this repo layers Luigi/LAW/
b2luigi/Hydra/Lightning, so the right layer for a new feature (task DAG, config schema, ETL, ML) is
often not obvious from the ask alone.

Use `.github/ISSUE_TEMPLATE/` when filing through GitHub — it has separate templates for bug
reports and feature requests.

## Code style and checks

Style is enforced by `pre-commit`, matching `pyproject.toml`:

```bash
pre-commit run --all-files   # black, isort, and the pre-commit-hooks set (fast, default stage)
flake8 .                      # manual stage — run explicitly, not part of default pre-commit
mypy .                         # manual stage — run explicitly, not part of default pre-commit
```

Note `flake8` and `mypy` hooks are configured with `stages: [manual]` in
`.pre-commit-config.yaml` — they do **not** run on a plain `git commit`, but CI's `lint` job runs
`pre-commit run --all-files` which does invoke them via the manual stage in CI. Run all four
locally before pushing so CI doesn't surprise you.

Conventions to follow (see `CLAUDE.md` for the architectural rationale behind these):

- Line length 120, `black`-formatted, `isort` with the `black` profile
- `mypy --disallow-untyped-defs`: all new functions need type annotations
- Backend-agnostic logic goes in `needle/tasks/base/`; `needle/tasks/law/` and
  `needle/tasks/b2luigi/` stay thin bindings. Never import across `law`/`b2luigi` within one DAG.
- No comments explaining *what* code does (names should do that); comments are reserved for
  non-obvious *why* (a workaround, an invariant, a subtlety a reader would trip on)
- Don't add abstractions, config knobs, or error handling for cases that can't currently occur

## Tests

```bash
pytest                 # default: excludes slow/law/b2luigi
pytest -m slow          # slow tests
pytest -m law            # law-backend task execution tests
pytest -m b2luigi        # b2luigi-backend task execution tests
```

- If you add or change behavior in `needle/tasks/law/` or `needle/tasks/b2luigi/`, add or update a
  test under `tests/trainings/` with the matching marker.
- If you add or change behavior in `needle/tasks/base/`, prefer a backend-agnostic test where
  possible so both bindings are exercised, or add coverage under both markers.
- Use the fixtures in `conftest.py` (`config_factory`, `make_parquet_file`, `fair_universe_sample`,
  `delphes_sample_root`/`delphes_sample_parquet`) rather than hand-rolling config/data setup.
  Env-gated fixtures skip cleanly when their env var is unset — don't make a test hard-fail because
  `FAIR_UNIVERSE_DATA`/`DELPHES_DATA_ROOT` isn't set in a given environment.
- Use `tmp_path` for anything under `tests/trainings/` to avoid collisions between concurrent runs.

## Docs

If your change affects the task DAG, config schema, or public API, update the relevant page under
`docs/concepts/` (`task_hierarchy.md`, `hydra_config.md`, `law_tasks.md`, `b2luigi_tasks.md`,
`downstream_tasks.md`, `lightning_and_hydra_integration.md`, `dask_awkward.md`) and/or docstrings —
the RST API reference under `docs/api/` is auto-generated from those.

Build docs locally to check before submitting a docs PR:

```bash
uv sync --group docs
uv run python -m sphinx -T -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

## Notes for AI agents working in this repo

- Read `CLAUDE.md` first — it has environment setup, architecture, and the current workspace
  layout. This file (`CONTRIBUTING.md`) governs workflow on top of that.
- Follow the branch naming and commit conventions above even when working from an auto-generated
  worktree branch name — rename the branch, or at minimum give the PR a title/description that
  follows the convention.
- Never force-push to `main`, rewrite shared branch history other people may have pulled, or merge
  your own PR without review unless explicitly instructed to.
- Run the full checklist at the top of this file before declaring a task complete; don't rely on
  partial checks (e.g. `black` alone) to stand in for the full `pre-commit` + `flake8` + `mypy` +
  `pytest` set.
- If a task spans both backends (`law` and `b2luigi`), implement and test both — don't add
  backend-specific behavior to `needle/tasks/base/` as a shortcut.
