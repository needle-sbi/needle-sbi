from __future__ import annotations

import argparse
import subprocess  # noqa: F401 -- re-exported so tests can patch `cli.subprocess.call`
import sys

# Deferred: importing needle.utils.logging pulls in the full `needle` package
# (needle.ml -> torch/lightning), which is slow to import. Argument parsing and
# tab-completion must stay fast, so this is only imported inside command
# functions that actually need the logger, never at module scope.
try:
    import argcomplete
except ImportError:
    argcomplete = None

_TASK_CHOICES = [
    "MainTask",
    "EstimatorTask",
    "SystematicTask",
    "EnsembleTask",
    "FoldTask",
    "TrainingTask",
    "DownstreamTask",
]


def _complete_task(**kwargs: object) -> list[str]:
    return _TASK_CHOICES


def cmd_init(args: argparse.Namespace) -> None:
    from needle.api.init import init

    init(args.directory, no_conf=args.no_conf, backend=getattr(args, "backend", "both"))


def cmd_run(args: argparse.Namespace) -> None:
    from needle.api.run import UnknownTaskError, run

    if args.backend == "law":
        config_file = getattr(args, "config_file", None)
        results_path = getattr(args, "results_path", None)
    else:
        config_file = getattr(args, "config_file", "conf/config.yaml")
        results_path = getattr(args, "results_path", "runs")

    try:
        result = run(
            task=args.task,
            backend=args.backend,
            config_file=config_file,
            results_path=results_path,
            batch_system=getattr(args, "batch_system", "local"),
            workers=getattr(args, "workers", 1),
            params=args.params,
        )
    except UnknownTaskError as e:
        raise SystemExit(str(e))

    if args.backend == "law":
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(prog="needle", description="NEEDLE CLI Manager")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser(
        "init",
        help="Initialize your project within NEEDLE. Adds the required templates",
    )
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
        choices=["law", "b2luigi", "both"],
        default="both",
        help="Workflow backend to scaffold (default: both)",
    )

    run = sub.add_parser("run", help="Run the NEEDLE training DAG")
    run_task_arg = run.add_argument(
        "task",
        nargs="?",
        default="MainTask",
        help="Task to run, e.g. MainTask, EstimatorTask, SystematicTask, EnsembleTask, FoldTask, "
        "DownstreamTask (default: MainTask)",
    )
    run_task_arg.completer = _complete_task  # type: ignore[attr-defined]
    run.add_argument(
        "--backend",
        choices=["law", "b2luigi"],
        default="law",
        help="Workflow backend to use (default: law)",
    )
    run.add_argument(
        "--config-file",
        dest="config_file",
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
        "--param",
        dest="params",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra parameter to pass to the selected task, e.g. --param estimator=my_estimator "
        "or --param downstream=my_downstream"
        "Can be given multiple times.",
    )

    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
