from __future__ import annotations

import argparse
import sys

# Deferred: importing needle.utils.logging pulls in the full `needle` package
# (needle.ml -> torch/lightning), which is slow to import. Argument parsing and
# tab-completion must stay fast, so this is only imported inside command
# functions that actually need the logger, never at module scope.
try:
    import argcomplete
except ImportError:
    argcomplete = None


def cmd_init(args: argparse.Namespace) -> None:
    from needle.api.init import init

    init(args.directory, no_conf=args.no_conf, backend=getattr(args, "backend", "both"))


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

    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
