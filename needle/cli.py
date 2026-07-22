from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_TEMPLATES = Path(__file__).parent / "templates"

_SETTINGS_JSON_TEMPLATE = """\
{
  "batch_system": "local",
  "htcondor_settings": {
    "request_memory": "2048MB",
    "request_cpus": 1,
    "+RequestRuntime": 3600
  },
  "slurm_settings": {
    "partition": "gpu",
    "time": "01:00:00",
    "mem": "4G",
    "cpus-per-task": 2
  }
}
"""


def _copy(src: Path, dst: Path, description: str) -> None:
    label = src.name
    if dst.exists():
        print(f"Skipped '{label}' ({description})")
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"Created '{label}' ({description})")


def cmd_init(args: argparse.Namespace) -> None:
    target = Path(args.directory).resolve()
    target.mkdir(parents=True, exist_ok=True)

    backend: str = getattr(args, "backend", "law")

    if backend == "law":
        _copy(
            src=_TEMPLATES / "law.cfg",
            dst=target / "law.cfg",
            description="LAW config file for managing Tasks",
        )
        _copy(
            src=_TEMPLATES / "index",
            dst=target / "index",
            description="Index of needle.law_tasks, update with `law index`",
        )
    elif backend == "b2luigi":
        settings_dst = target / "settings.json"
        if settings_dst.exists():
            print("Skipped 'settings.json' (b2luigi settings file)")
        else:
            settings_dst.write_text(_SETTINGS_JSON_TEMPLATE)
            print("Created 'settings.json' (b2luigi settings file)")

    setup_dst = target / "setup.sh"
    _copy(
        src=_TEMPLATES / "setup.sh",
        dst=setup_dst,
        description="Setup script for setting up the NEEDLE environment",
    )
    if setup_dst.exists():
        setup_dst.chmod(0o755)

    if not args.no_conf:
        _copy(
            src=_TEMPLATES / "conf",
            dst=target / "conf",
            description="Config directory following the hydra schema",
        )


def cmd_run(args: argparse.Namespace) -> None:
    backend: str = args.backend

    if backend == "law":
        task_name = getattr(args, "task", "MainTask")
        config_file = getattr(args, "config", None)
        results_path = getattr(args, "results_path", None)

        law_args = ["law", "run", task_name]
        if config_file:
            law_args += ["--config-file", config_file]
        if results_path:
            law_args += ["--results-path", results_path]

        sys.exit(subprocess.call(law_args))

    elif backend == "b2luigi":
        import b2luigi

        from needle.tasks.b2luigi import MainTask
        from needle.tasks.b2luigi.workflows.common import configure_b2luigi

        config_file = getattr(args, "config", "conf/config.yaml")
        results_path = getattr(args, "results_path", "runs")
        batch_system = getattr(args, "batch_system", "local")
        workers = getattr(args, "workers", 1)

        configure_b2luigi(results_path=results_path, batch_system=batch_system)

        b2luigi.process(
            MainTask(
                config_file=config_file,
                results_path=results_path,
            ),
            workers=workers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="needle", description="NEEDLE CLI Manager")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- needle init ---
    init = sub.add_parser("init", help="Initialize your project within NEEDLE. Adds the required templates")
    init.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory (default: current working directory)",
    )
    init.add_argument(
        "--no-conf",
        action="store_true",
        help="Skip creating the conf/ directory with default Hydra config groups",
    )
    init.add_argument(
        "--backend",
        choices=["law", "b2luigi"],
        default="law",
        help="Workflow backend to scaffold (default: law)",
    )

    # --- needle run ---
    run = sub.add_parser("run", help="Run the NEEDLE training DAG")
    run.add_argument(
        "--backend",
        choices=["law", "b2luigi"],
        default="law",
        help="Workflow backend to use (default: law)",
    )
    run.add_argument(
        "--config",
        default="conf/config.yaml",
        help="Path to the Hydra config file (default: conf/config.yaml)",
    )
    run.add_argument(
        "--results-path",
        default="runs",
        dest="results_path",
        help="Root directory for results (default: runs)",
    )
    run.add_argument(
        "--batch-system",
        default="local",
        dest="batch_system",
        choices=["local", "htcondor", "slurm", "lsf"],
        help="Batch system for b2luigi backend (default: local)",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for b2luigi (default: 1)",
    )
    run.add_argument(
        "--task",
        default="MainTask",
        help="LAW task name to run (default: MainTask, law backend only)",
    )

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
