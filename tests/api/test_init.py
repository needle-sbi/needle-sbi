"""Tests for needle/api/init.py: init(), InitResult.

Mirrors tests/test_cli.py::TestCmdInit but calling needle.api.init.init() directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from needle.api.init import init


def test_law_backend_creates_law_files(tmp_path: Path) -> None:
    result = init(tmp_path, no_conf=False, backend="law")

    assert (tmp_path / "law.cfg").exists()
    assert (tmp_path / "index").exists()
    assert (tmp_path / "setup.sh").exists()
    assert (tmp_path / "conf").is_dir()
    assert not (tmp_path / "settings.json").exists()
    assert (tmp_path / "law.cfg") in result.created


def test_b2luigi_backend_creates_settings_json(tmp_path: Path) -> None:
    init(tmp_path, no_conf=False, backend="b2luigi")

    assert (tmp_path / "settings.json").exists()
    assert (tmp_path / "setup.sh").exists()
    assert not (tmp_path / "law.cfg").exists()
    assert not (tmp_path / "index").exists()


def test_no_conf_skips_conf_directory(tmp_path: Path) -> None:
    init(tmp_path, no_conf=True, backend="law")
    assert not (tmp_path / "conf").exists()


def test_setup_sh_is_executable(tmp_path: Path) -> None:
    init(tmp_path, no_conf=True, backend="law")
    mode = (tmp_path / "setup.sh").stat().st_mode
    assert mode & 0o111


def test_rerun_skips_existing_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    init(tmp_path, no_conf=False, backend="law")
    capsys.readouterr()

    result = init(tmp_path, no_conf=False, backend="law")
    out = capsys.readouterr().out
    assert "Skipped 'law.cfg'" in out
    assert "Skipped 'setup.sh'" in out
    assert "Skipped 'conf'" in out
    assert (tmp_path / "law.cfg") in result.skipped


def test_creates_target_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "project"
    init(target, no_conf=True, backend="law")
    assert target.is_dir()


def test_verbose_false_suppresses_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    init(tmp_path, no_conf=True, backend="law", verbose=False)
    out = capsys.readouterr().out
    assert out == ""
