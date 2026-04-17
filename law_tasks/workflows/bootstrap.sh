#!/usr/bin/env bash

# Sources:
#   Original: https://github.com/riga/law/blob/master/examples/htcondor_at_cern/analysis/bootstrap.sh
#   Adapted by K. Schmidt using Claude Sonnet 4.6

# Description:
#   This file is ran by law directly after starting the job. Used to install the dependencies (either
#   with uv or pip) and source the .venv as well as the setup.sh script.

action() {
    # Use user environment variables or set local path relative to $LAW_JOB_HOME
    # double curly brackets implies runtime variable rendering by law
    _RENDERED_SCRIPT_DIR="{{script_dir}}"
    SCRIPT_DIR="${SCRIPT_DIR:-$_RENDERED_SCRIPT_DIR}"
    UV_INSTALL_DIR="${UV_INSTALL_DIR:-$LAW_JOB_HOME/.local/bin}"
    UV_CACHE_DIR="${UV_CACHE_DIR:-$LAW_JOB_HOME/.uv_cache}"
    PIP_CACHE_DIR="${PIP_CACHE_DIR:-$LAW_JOB_HOME/.cache/pip}"
    mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

    cd "$SCRIPT_DIR"

    # Install astral uv for dependency management
    curl -LsSf "https://astral.sh/uv/install.sh" | sh
    source "$UV_INSTALL_DIR"

    uv python pin 3.12  # TODO Support other python versions
    uv sync --no-dev --no-install-project

    # Source the venv (either with uv or pip as a fallback)
    if [ $? -ne 0 ]; then
        echo "uv sync failed, falling back to pip"
        "$uv" run python -m venv "$LAW_JOB_HOME/.venv"
        source "$LAW_JOB_HOME/.venv/bin/activate"
        pip install --quiet --no-cache-dir -e .
    else
        source ".venv/bin/activate"
    fi

    source setup.sh
}
action
