"""Tests for needle/cli.py: `needle init` and `needle run` (both backends).

Fast, hermetic tests - no real training, no real `law run`/`b2luigi.process()`
execution. Subprocess/`b2luigi.process` calls are mocked so these run as part of
the default (non-slow, non-law, non-b2luigi) test suite. End-to-end CLI-level
compatibility with `law run` and `python3 -m luigi` is covered separately in
``tests/trainings/test_cli_compat.py``.
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
        assert args.backend == "law"

    def test_main_parses_init_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_cmd_init = MagicMock(return_value=None)
        monkeypatch.setattr(cli, "cmd_init", mock_cmd_init)

        _run_main(["init", "my_project", "--no-conf", "--backend", "b2luigi"])

        args = mock_cmd_init.call_args.args[0]
        assert args.directory == "my_project"
        assert args.no_conf is True
        assert args.backend == "b2luigi"


# ---------------------------------------------------------------------------
# `needle run` — argument parsing
# ---------------------------------------------------------------------------


class TestCmdRunArgParsing:
    def test_main_parses_run_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_cmd_run = MagicMock(return_value=None)
        monkeypatch.setattr(cli, "cmd_run", mock_cmd_run)

        _run_main(["run"])

        args = mock_cmd_run.call_args.args[0]
        assert args.task == "MainTask"
        assert args.backend == "law"
        assert args.config_file == "conf/config.yaml"
        assert args.results_path == "runs"
        assert args.batch_system == "local"
        assert args.workers == 1
        assert args.params == []

    def test_main_parses_run_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_cmd_run = MagicMock(return_value=None)
        monkeypatch.setattr(cli, "cmd_run", mock_cmd_run)

        _run_main(
            [
                "run",
                "EnsembleTask",
                "--backend",
                "b2luigi",
                "--config-file",
                "my/config.yaml",
                "--results-path",
                "my_runs",
                "--batch-system",
                "htcondor",
                "--workers",
                "4",
                "--param",
                "estimator=model_A",
                "--param",
                "systematic=nominal",
            ]
        )

        args = mock_cmd_run.call_args.args[0]
        assert args.task == "EnsembleTask"
        assert args.backend == "b2luigi"
        assert args.config_file == "my/config.yaml"
        assert args.results_path == "my_runs"
        assert args.batch_system == "htcondor"
        assert args.workers == 4
        assert args.params == ["estimator=model_A", "systematic=nominal"]

    def test_run_rejects_unknown_batch_system(self) -> None:
        old_argv = sys.argv
        sys.argv = ["needle", "run", "--batch-system", "bogus"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 2
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# `needle run --backend law` — builds and forwards the correct `law run` argv
# ---------------------------------------------------------------------------


class TestCmdRunLawBackend:
    def test_builds_law_run_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_call = MagicMock(return_value=0)
        monkeypatch.setattr(cli.subprocess, "call", mock_call)

        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_run(
                argparse.Namespace(
                    task="EnsembleTask",
                    backend="law",
                    config_file="conf/config.yaml",
                    results_path="runs",
                    params=["estimator=model_A", "systematic=nominal"],
                )
            )

        assert exc_info.value.code == 0
        called_argv = mock_call.call_args.args[0]
        assert called_argv[:3] == ["law", "run", "EnsembleTask"]
        assert "--config-file" in called_argv
        assert called_argv[called_argv.index("--config-file") + 1] == "conf/config.yaml"
        assert "--results-path" in called_argv
        assert called_argv[called_argv.index("--results-path") + 1] == "runs"
        assert "--estimator" in called_argv
        assert called_argv[called_argv.index("--estimator") + 1] == "model_A"
        assert "--systematic" in called_argv
        assert called_argv[called_argv.index("--systematic") + 1] == "nominal"

    def test_underscore_params_become_dashed_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_call = MagicMock(return_value=0)
        monkeypatch.setattr(cli.subprocess, "call", mock_call)

        with pytest.raises(SystemExit):
            cli.cmd_run(
                argparse.Namespace(
                    task="MainTask",
                    backend="law",
                    config=None,
                    results_path=None,
                    params=["strict_config=RAISE"],
                )
            )

        called_argv = mock_call.call_args.args[0]
        assert "--strict-config" in called_argv
        assert called_argv[called_argv.index("--strict-config") + 1] == "RAISE"

    def test_valueless_param_is_passed_as_bare_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_call = MagicMock(return_value=0)
        monkeypatch.setattr(cli.subprocess, "call", mock_call)

        with pytest.raises(SystemExit):
            cli.cmd_run(
                argparse.Namespace(
                    task="MainTask",
                    backend="law",
                    config_file=None,
                    results_path=None,
                    params=["print-deps", "estimator=model_A"],
                )
            )

        called_argv = mock_call.call_args.args[0]
        assert "--print-deps" in called_argv
        # A bare flag must not swallow the next `--param`'s value as its own.
        idx = called_argv.index("--print-deps")
        assert called_argv[idx + 1] == "--estimator"

    def test_propagates_subprocess_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli.subprocess, "call", MagicMock(return_value=1))

        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_run(
                argparse.Namespace(
                    task="MainTask",
                    backend="law",
                    config=None,
                    results_path=None,
                    params=[],
                )
            )
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# `needle run --backend b2luigi` — constructs the task and drives it through
# b2luigi.process(), guarding sys.argv around the call.
# ---------------------------------------------------------------------------


class TestCmdRunB2luigiBackend:
    def test_instantiates_task_and_calls_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_process = MagicMock()
        mock_configure = MagicMock()
        monkeypatch.setattr("b2luigi.process", mock_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", mock_configure)

        cli.cmd_run(
            argparse.Namespace(
                task="EnsembleTask",
                backend="b2luigi",
                config="conf/config.yaml",
                results_path="runs",
                batch_system="local",
                workers=2,
                params=["estimator=model_A", "systematic=nominal"],
            )
        )

        mock_configure.assert_called_once_with(results_path="runs", batch_system="local")
        assert mock_process.call_count == 1
        (task_instance,), kwargs = mock_process.call_args
        assert type(task_instance).__name__ == "EnsembleTask"
        assert task_instance.estimator == "model_A"
        assert task_instance.systematic == "nominal"
        assert kwargs == {"workers": 2, "batch": False}

    def test_valueless_param_becomes_boolean_true_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_process = MagicMock()
        monkeypatch.setattr("b2luigi.process", mock_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", MagicMock())

        mock_task_cls = MagicMock()
        monkeypatch.setattr("needle.tasks.b2luigi.FakeTask", mock_task_cls, raising=False)

        cli.cmd_run(
            argparse.Namespace(
                task="FakeTask",
                backend="b2luigi",
                config_file="conf/config.yaml",
                results_path="runs",
                batch_system="local",
                workers=1,
                params=["dry_run"],
            )
        )

        mock_task_cls.assert_called_once_with(config_file="conf/config.yaml", results_path="runs", dry_run=True)

    def test_batch_system_other_than_local_sets_batch_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_process = MagicMock()
        monkeypatch.setattr("b2luigi.process", mock_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", MagicMock())

        cli.cmd_run(
            argparse.Namespace(
                task="MainTask",
                backend="b2luigi",
                config="conf/config.yaml",
                results_path="runs",
                batch_system="htcondor",
                workers=1,
                params=[],
            )
        )

        _, kwargs = mock_process.call_args
        assert kwargs["batch"] is True

    def test_restores_sys_argv_after_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_argv_during_call = []

        def fake_process(task, **kwargs):
            captured_argv_during_call.append(list(sys.argv))

        monkeypatch.setattr("b2luigi.process", fake_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", MagicMock())

        original_argv = ["needle", "run", "MainTask", "--backend", "b2luigi"]
        monkeypatch.setattr(sys, "argv", original_argv)

        cli.cmd_run(
            argparse.Namespace(
                task="MainTask",
                backend="b2luigi",
                config="conf/config.yaml",
                results_path="runs",
                batch_system="local",
                workers=1,
                params=[],
            )
        )

        assert captured_argv_during_call == [["needle"]]
        assert sys.argv == original_argv

    def test_restores_sys_argv_even_if_process_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raising_process(task, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("b2luigi.process", raising_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", MagicMock())

        original_argv = ["needle", "run", "MainTask", "--backend", "b2luigi"]
        monkeypatch.setattr(sys, "argv", original_argv)

        with pytest.raises(RuntimeError):
            cli.cmd_run(
                argparse.Namespace(
                    task="MainTask",
                    backend="b2luigi",
                    config="conf/config.yaml",
                    results_path="runs",
                    batch_system="local",
                    workers=1,
                    params=[],
                )
            )

        assert sys.argv == original_argv

    def test_unknown_task_raises_system_exit_listing_available(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_run(
                argparse.Namespace(
                    task="DoesNotExistTask",
                    backend="b2luigi",
                    config="conf/config.yaml",
                    results_path="runs",
                    batch_system="local",
                    workers=1,
                    params=[],
                )
            )
        assert "DoesNotExistTask" in str(exc_info.value)
        assert "MainTask" in str(exc_info.value)
