#!/usr/bin/env bash

# Sources:
#   Original: https://github.com/riga/law/blob/master/examples/htcondor_at_cern/analysis/bootstrap.sh
#   Adapted by K. Schmidt using Claude Sonnet 4.6

# Description:
#   This file is ran by law directly after starting the job. Used to install the dependencies (either
#   with uv or pip) and source the .venv as well as the setup.sh script.

action() {
    # Resolve hashed filenames by glob
    local pyproject
    pyproject=$(ls "$LAW_JOB_HOME"/pyproject_*.toml 2>/dev/null | head -1)
    local setup
    setup=$(ls "$LAW_JOB_HOME"/setup_*.sh 2>/dev/null | head -1)
    export UV_CACHE_DIR="$LAW_JOB_HOME"/.uv_cache

    # Install astral uv for dependency management
    curl -LsSf "https://astral.sh/uv/install.sh" | sh
    source $HOME/.local/bin

    # Copy hashed pyproject.toml to a clean working dir
    cp "$pyproject" "$LAW_JOB_HOME/pyproject.toml"

    uv python pin 3.12  # TODO Support other python versions
    uv sync --no-dev --no-install-project

    # Source the venv (either with uv or pip as a fallback)
    if [ $? -ne 0 ]; then
        echo "uv sync failed, falling back to pip"
        "$uv" run python -m venv "$LAW_JOB_HOME/.venv"
        source "$LAW_JOB_HOME/.venv/bin/activate"
        pip install --quiet --no-cache-dir -e "$LAW_JOB_HOME"
    else
        source "$LAW_JOB_HOME/.venv/bin/activate"
    fi

    # Run setup script
    [ -n "$setup" ] && source "$setup"
}
action
