## Summary

What changed and why. Link the issue if there is one (`Closes #N` / `Refs #N`).

## Backend(s) affected

- [ ] law
- [ ] b2luigi
- [ ] both
- [ ] neither / general

## Test plan

- [ ] `pre-commit run --all-files`
- [ ] `flake8 .`
- [ ] `mypy .`
- [ ] `pytest`
- [ ] `pytest -m slow` (if applicable)
- [ ] `pytest -m law` (if `needle/tasks/law/` touched)
- [ ] `pytest -m b2luigi` (if `needle/tasks/b2luigi/` touched)
- [ ] Manual check (e.g. `law run ...` / `b2luigi run ...`), if applicable:

## Docs

- [ ] Docs updated (`docs/concepts/...`, docstrings) — or not applicable
