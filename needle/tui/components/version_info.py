#!/usr/bin/env python3
"""Find and display the package version for the TUI.
Disclaimer: Generated using Claude 4.6
"""

import json
import os
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_python_version() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_package_version(package_name: str) -> str:
    """Get installed package version or 'N/A' if not found."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "N/A"


def _pyproject_path() -> Path:
    return Path(__file__).parent.parent.parent.parent / "pyproject.toml"


def get_needle_version() -> str:
    """Get NEEDLE version from pyproject.toml."""
    pip_version = get_package_version("needle-sbi")
    if pip_version != "N/A":
        return pip_version
    try:
        pyproject_path = _pyproject_path()

        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            return pyproject.get("project", {}).get("version", "N/A")
        return "N/A"
    except Exception:
        return "N/A"


def get_cuda_status() -> tuple[bool, str | None]:
    """Return (is_available, cuda_version). cuda_version is None if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.version.cuda
        return False, None
    except Exception:
        return False, None


def get_project_files_status() -> dict[str, bool]:
    """Check whether project scaffold files exist relative to the active NEEDLE project."""
    base = Path(os.environ.get("SCRIPT_DIR", os.getcwd()))
    return {
        "law.cfg": (base / "law.cfg").is_file(),
        "settings.json": (base / "settings.json").is_file(),
    }


def get_all_versions() -> dict:
    """Get all version information."""
    cuda_available, cuda_version = get_cuda_status()
    return {
        "python": get_python_version(),
        "needle": get_needle_version(),
        "law": get_package_version("law"),
        "b2luigi": get_package_version("b2luigi"),
        "lightning": get_package_version("lightning"),
        "pytorch": get_package_version("torch"),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "project_files": get_project_files_status(),
    }


def format_versions_as_text() -> list[str]:
    """Return version information as list of formatted strings."""
    versions = get_all_versions()
    files = versions["project_files"]
    cuda_text = f"CUDA: {versions['cuda_version']}" if versions["cuda_available"] else "CUDA: Not available"
    return [
        f"Python:    {versions['python']}",
        f"NEEDLE:    {versions['needle']}",
        f"Law:       {versions['law']}",
        f"b2luigi:   {versions['b2luigi']}",
        f"Lightning: {versions['lightning']}",
        f"PyTorch:   {versions['pytorch']}",
        cuda_text,
        f"law.cfg: {'found' if files['law.cfg'] else 'missing'}",
        f"settings.json: {'found' if files['settings.json'] else 'missing'}",
    ]


def format_version_panels() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (left_lines, right_lines) as (text, color) pairs for the two-column banner panel.

    color is one of: none, red, green, blue, orange.
    Left panel: package versions. Right panel: environment/project status checks.
    """
    versions = get_all_versions()
    files = versions["project_files"]

    cuda_text = f"CUDA: {versions['cuda_version']}" if versions["cuda_available"] else "CUDA: Not available"
    cuda_color = "green" if versions["cuda_available"] else "red"

    def _file_line(name: str, found: bool) -> tuple[str, str]:
        mark = "✓" if found else "✗"
        return f"{name} {mark}", ("green" if found else "red")

    left_lines = [
        (f"Python:    {versions['python']}", "none"),
        (f"NEEDLE:    {versions['needle']}", "none"),
        (f"Law:       {versions['law']}", "none"),
        (f"b2luigi:   {versions['b2luigi']}", "none"),
        (f"Lightning: {versions['lightning']}", "none"),
        (f"PyTorch:   {versions['pytorch']}", "none"),
    ]
    right_lines = [
        (cuda_text, cuda_color),
        _file_line("law.cfg", files["law.cfg"]),
        _file_line("settings.json", files["settings.json"]),
    ]
    return left_lines, right_lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get NEEDLE version information")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--text", action="store_true", help="Output as text lines (default)")
    parser.add_argument(
        "--panel-lines",
        action="store_true",
        help="Output 'side<0x1f>text<0x1f>color' lines (side: L/R) for shell consumption",
    )
    args = parser.parse_args()

    if args.json:
        print(json.dumps(get_all_versions()))
    elif args.panel_lines:
        left_lines, right_lines = format_version_panels()
        for text, color in left_lines:
            print(f"L\x1f{text}\x1f{color}")
        for text, color in right_lines:
            print(f"R\x1f{text}\x1f{color}")
    else:
        # Default to text format
        for line in format_versions_as_text():
            print(line)
