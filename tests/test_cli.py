"""Tests for needle/cli.py: `needle init`.

Fast, hermetic tests - no real training, no filesystem side effects beyond
``tmp_path``. ``needle`` no longer wraps task execution (``law run``/``b2luigi run``
are the recommended entry points directly, or ``needle.api.run()`` from Python).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from needle import cli


def _run_main(args: list[str]) -> None:
    """Invoke needle.cli.main() with a given argv, as the console script would."""
    old_argv = sys.argv
    sys.argv = ["needle", *args]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# `needle init`
# ---------------------------------------------------------------------------


class TestCmdInit:
    def test_law_backend_creates_law_files(self, tmp_path: Path) -> None:
        cli.cmd_init(argparse.Namespace(directory=str(tmp_path), no_conf=False, backend="law"))

        assert (tmp_path / "law.cfg").exists()
        assert (tmp_path / "index").exists()
        assert (tmp_path / "setup.sh").exists()
        assert (tmp_path / "conf").is_dir()
        assert not (tmp_path / "settings.json").exists()

    def test_b2luigi_backend_creates_settings_json(self, tmp_path: Path) -> None:
        cli.cmd_init(argparse.Namespace(directory=str(tmp_path), no_conf=False, backend="b2luigi"))

        assert (tmp_path / "settings.json").exists()
        assert (tmp_path / "tasks.py").exists()
        assert (tmp_path / "setup.sh").exists()
        assert not (tmp_path / "law.cfg").exists()
        assert not (tmp_path / "index").exists()

    def test_no_conf_skips_conf_directory(self, tmp_path: Path) -> None:
        cli.cmd_init(argparse.Namespace(directory=str(tmp_path), no_conf=True, backend="law"))

        assert not (tmp_path / "conf").exists()

    def test_setup_sh_is_executable(self, tmp_path: Path) -> None:
        cli.cmd_init(argparse.Namespace(directory=str(tmp_path), no_conf=True, backend="law"))

        mode = (tmp_path / "setup.sh").stat().st_mode
        assert mode & 0o111

    def test_rerun_skips_existing_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cli.cmd_init(argparse.Namespace(directory=str(tmp_path), no_conf=False, backend="law"))
        capsys.readouterr()

        cli.cmd_init(argparse.Namespace(directory=str(tmp_path), no_conf=False, backend="law"))
        out = capsys.readouterr().out
        assert "Skipped 'law.cfg'" in out
        assert "Skipped 'setup.sh'" in out
        assert "Skipped 'conf'" in out

    def test_creates_target_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "project"
        cli.cmd_init(argparse.Namespace(directory=str(target), no_conf=True, backend="law"))
        assert target.is_dir()

    def test_main_parses_init_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        mock_cmd_init = MagicMock(return_value=None)
        monkeypatch.setattr(cli, "cmd_init", mock_cmd_init)

        _run_main(["init"])

        args = mock_cmd_init.call_args.args[0]
        assert args.directory == "."
        assert args.no_conf is False
        assert args.backend == "both"

    def test_main_parses_init_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_cmd_init = MagicMock(return_value=None)
        monkeypatch.setattr(cli, "cmd_init", mock_cmd_init)

        _run_main(["init", "my_project", "--no-conf", "--backend", "b2luigi"])

        args = mock_cmd_init.call_args.args[0]
        assert args.directory == "my_project"
        assert args.no_conf is True
        assert args.backend == "b2luigi"
