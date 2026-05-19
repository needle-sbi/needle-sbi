#!/usr/bin/env python3
"""Find and display the package version for the TUI.
Disclaimer: Generated using Claude 4.6
"""

import json
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


def get_needle_version() -> str:
    """Get NEEDLE version from pyproject.toml."""
    pip_version = get_package_version("needle")
    if pip_version != "N/A":
        return pip_version
    try:
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            return pyproject.get("project", {}).get("version", "N/A")
        return "N/A"
    except Exception:
        return "N/A"


def get_all_versions() -> dict:
    """Get all version information."""
    return {
        "python": get_python_version(),
        "needle": get_needle_version(),
        "law": get_package_version("law"),
        "lightning": get_package_version("lightning"),
        "pytorch": get_package_version("torch"),
    }


def format_versions_as_text() -> list[str]:
    """Return version information as list of formatted strings."""
    versions = get_all_versions()
    return [
        f"Python:    {versions['python']}",
        f"NEEDLE:    {versions['needle']}",
        f"Law:       {versions['law']}",
        f"Lightning: {versions['lightning']}",
        f"PyTorch:   {versions['pytorch']}",
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get NEEDLE version information")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--text", action="store_true", help="Output as text lines (default)")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(get_all_versions()))
    else:
        # Default to text format
        for line in format_versions_as_text():
            print(line)
