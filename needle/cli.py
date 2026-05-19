import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TEMPLATES = Path(__file__).parent / "templates"


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


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.directory).resolve()
    target.mkdir(parents=True, exist_ok=True)

    _copy(
        src=_TEMPLATES / "law.cfg",
        dst=target / "law.cfg",
        description="LAW config file for managing Tasks",
    )
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

    law_home = target / ".law"
    law_home.mkdir(exist_ok=True)
    env = {**os.environ, "LAW_HOME": str(law_home), "LAW_CONFIG_FILE": str(target / "law.cfg")}
    result = subprocess.run(["law", "index"], env=env, cwd=target, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout, end="\r")
        print("Next, source the `setup.sh` script within your virtual environment")
    else:
        print(result.stderr)

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(prog="needle", description="NEEDLE CLI Manager")
    sub = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args()
    if args.command == "init":
        sys.exit(cmd_init(args))
