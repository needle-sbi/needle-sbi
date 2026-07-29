"""Tests for needle/api/run.py: run(), RunResult, UnknownTaskError.

Fast, hermetic tests mirroring tests/test_cli.py's law/b2luigi assertions but
calling needle.api.run.run() directly instead of through the CLI.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from needle.api.run import RunResult, UnknownTaskError, run


class TestRunLawBackend:
    def test_builds_law_run_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_call = MagicMock(return_value=0)
        monkeypatch.setattr("subprocess.call", mock_call)

        result = run(
            "EnsembleTask",
            backend="law",
            config_file="conf/config.yaml",
            results_path="runs",
            params=["estimator=model_A", "systematic=nominal"],
        )

        assert result == RunResult(returncode=0)
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

    def test_accepts_dict_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_call = MagicMock(return_value=0)
        monkeypatch.setattr("subprocess.call", mock_call)

        run("MainTask", backend="law", params={"estimator": "model_A", "print_deps": True})

        called_argv = mock_call.call_args.args[0]
        assert "--estimator" in called_argv
        assert called_argv[called_argv.index("--estimator") + 1] == "model_A"
        assert called_argv[-1] == "--print-deps"

    def test_omits_flags_when_config_and_results_path_are_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_call = MagicMock(return_value=0)
        monkeypatch.setattr("subprocess.call", mock_call)

        run("MainTask", backend="law", config_file=None, results_path=None, params=[])

        called_argv = mock_call.call_args.args[0]
        assert "--config-file" not in called_argv
        assert "--results-path" not in called_argv

    def test_valueless_param_is_passed_as_bare_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_call = MagicMock(return_value=0)
        monkeypatch.setattr("subprocess.call", mock_call)

        run("MainTask", backend="law", params=["print-deps", "estimator=model_A"])

        called_argv = mock_call.call_args.args[0]
        assert "--print-deps" in called_argv
        idx = called_argv.index("--print-deps")
        assert called_argv[idx + 1] == "--estimator"

    def test_propagates_subprocess_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("subprocess.call", MagicMock(return_value=1))

        result = run("MainTask", backend="law", params=[])
        assert result.returncode == 1


class TestRunB2luigiBackend:
    def test_instantiates_task_and_calls_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_process = MagicMock()
        mock_configure = MagicMock()
        monkeypatch.setattr("b2luigi.process", mock_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", mock_configure)

        result = run(
            "EnsembleTask",
            backend="b2luigi",
            config_file="conf/config.yaml",
            results_path="runs",
            batch_system="local",
            workers=2,
            params=["estimator=model_A", "systematic=nominal"],
        )

        assert result == RunResult(returncode=None)
        mock_configure.assert_called_once_with(batch_system="local")
        assert mock_process.call_count == 1
        (task_instance,), kwargs = mock_process.call_args
        assert type(task_instance).__name__ == "EnsembleTask"
        assert task_instance.estimator == "model_A"
        assert task_instance.systematic == "nominal"
        assert kwargs == {"workers": 2, "batch": False}

    def test_accepts_dict_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_process = MagicMock()
        monkeypatch.setattr("b2luigi.process", mock_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", MagicMock())
        mock_task_cls = MagicMock()
        monkeypatch.setattr("needle.tasks.b2luigi.FakeTask", mock_task_cls, raising=False)

        run(
            "FakeTask", backend="b2luigi", config_file="conf/config.yaml", results_path="runs", params={"dry_run": True}
        )

        mock_task_cls.assert_called_once_with(config_file="conf/config.yaml", results_path="runs", dry_run=True)

    def test_valueless_param_becomes_boolean_true_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_process = MagicMock()
        monkeypatch.setattr("b2luigi.process", mock_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", MagicMock())
        mock_task_cls = MagicMock()
        monkeypatch.setattr("needle.tasks.b2luigi.FakeTask", mock_task_cls, raising=False)

        run("FakeTask", backend="b2luigi", config_file="conf/config.yaml", results_path="runs", params=["dry_run"])

        mock_task_cls.assert_called_once_with(config_file="conf/config.yaml", results_path="runs", dry_run=True)

    def test_batch_system_other_than_local_sets_batch_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_process = MagicMock()
        monkeypatch.setattr("b2luigi.process", mock_process)
        monkeypatch.setattr("needle.tasks.b2luigi.workflows.common.configure_b2luigi", MagicMock())

        run("MainTask", backend="b2luigi", batch_system="htcondor", params=[])

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

        run("MainTask", backend="b2luigi", params=[])

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
            run("MainTask", backend="b2luigi", params=[])

        assert sys.argv == original_argv

    def test_unknown_task_raises_unknown_task_error(self) -> None:
        with pytest.raises(UnknownTaskError) as exc_info:
            run("DoesNotExistTask", backend="b2luigi", params=[])

        assert "DoesNotExistTask" in str(exc_info.value)
        assert "MainTask" in str(exc_info.value)
