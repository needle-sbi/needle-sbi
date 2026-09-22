"""Auto-derive the `--benchmark-json` output path for `tests/benchmarks/`."""

import re
from pathlib import Path
from typing import Optional, Tuple

import pytest

RESULTS_DIR = Path(__file__).parent / "results"

# needle -> (stem, split_by_marker). `split_by_marker=True` writes separate `<stem>_fast.json` /
# `<stem>_slow.json` files depending on the active `-m` expression; `False` always writes a single
# `<stem>.json` (used by suites where every parametrization is marked `slow`, so there is only ever
# one meaningful run).
_STEMS: dict = {
    "test_ingestion_profiling": ("ingestion_profiling", False),
    "test_root_vs_parquet": ("root_vs_parquet", True),
}


def _is_slow_run(config: pytest.Config) -> bool:
    markexpr = config.option.markexpr or ""
    return bool(re.search(r"(?<!not )\bslow\b", markexpr))


def _resolve_stem(config: pytest.Config) -> Optional[Tuple[str, bool]]:
    """Match the invoked test file(s) to a known benchmark suite stem, if any."""
    invocation = " ".join(str(arg) for arg in config.args)
    for needle, entry in _STEMS.items():
        if needle in invocation:
            return entry
    return None


def pytest_configure(config: pytest.Config) -> None:
    if config.option.benchmark_json is not None:
        # The caller passed an explicit --benchmark-json; respect it.
        return
    if not config.getoption("benchmark_only"):
        # Only auto-derive a path for dedicated benchmark runs (--benchmark-only), so plain
        # `pytest` runs (where benchmarks are skipped) never open/leave behind an empty file.
        return

    resolved = _resolve_stem(config)
    if resolved is None:
        return
    stem, split_by_marker = resolved

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if split_by_marker:
        suffix = "slow" if _is_slow_run(config) else "fast"
        output_path = RESULTS_DIR / f"{stem}_{suffix}.json"
    else:
        output_path = RESULTS_DIR / f"{stem}.json"
    config.option.benchmark_json = output_path.open("wb")
