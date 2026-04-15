#!/usr/bin/env bash

# Sources:
#   Original: https://github.com/riga/law/blob/master/examples/htcondor_at_cern/analysis/bootstrap.sh
#   Adapted by K. Schmidt using Claude Sonnet 4.6

action() {
    # resolve hashed filenames by glob
    local pyproject
    pyproject=$(ls "$LAW_JOB_HOME"/pyproject_*.toml 2>/dev/null | head -1)
    local setup
    setup=$(ls "$LAW_JOB_HOME"/setup_*.sh 2>/dev/null | head -1)

    curl -LsSf "https://astral.sh/uv/install.sh" | sh
    source $HOME/.local/bin

    # copy pyproject.toml to a clean working dir so uv sees a proper project root
    cp "$pyproject" "$LAW_JOB_HOME/pyproject.toml"

    uv python pin 3.12  # TODO Support other python versions
    uv sync --no-dev --no-install-project

    if [ $? -ne 0 ]; then
        echo "uv sync failed, falling back to pip"
        "$uv" run python -m venv "$LAW_JOB_HOME/.venv"
        source "$LAW_JOB_HOME/.venv/bin/activate"
        pip install --quiet --no-cache-dir -e "$LAW_JOB_HOME"
    else
        source "$LAW_JOB_HOME/.venv/bin/activate"
    fi

    [ -n "$setup" ] && source "$setup"
}
action
