#!/bin/bash

ENV_NAME="NEEDLE"

# prefixed with underscore and cleaned up at end to avoid leaking into env
_NEEDLE_RED='\033[0;31m'
_NEEDLE_GREEN='\033[0;32m'
_NEEDLE_ORANGE='\033[0;33m'
_NEEDLE_NC='\033[0m' # No Color
_NEEDLE_BLUE='\033[0;34m'

# guard: `source setup.sh` works, `bash setup.sh` does not
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] 2>/dev/null; then
    echo -e "\033[0;31mError: This script must be sourced, not executed. Run: source setup.sh\033[0m"
    exit 1
fi
if [[ -n "$ZSH_EVAL_CONTEXT" ]] && [[ "$ZSH_EVAL_CONTEXT" != *:file* ]]; then
    echo -e "\033[0;31mError: This script must be sourced, not executed. Run: source setup.sh\033[0m"
    exit 1
fi

# resolve script dir (needed later for LAW_HOME)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"

# Check if running in remote HTCondor job (LAW sets these variables)
if [[ -n "$LAW_HTCONDOR_JOB_NUMBER" ]] || [[ -n "$LAW_JOB_INIT_DIR" ]]; then
    echo -e "${_NEEDLE_BLUE}Running in remote HTCondor context, activating virtual environment...${_NEEDLE_NC}"
    # Try to activate venv if it exists
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    elif [[ -f "$SCRIPT_DIR/../.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/../.venv/bin/activate"
    else
        echo -e "${_NEEDLE_ORANGE}Warning: Virtual environment not found, assuming packages are available system-wide${_NEEDLE_NC}"
    fi
else
    # Local execution - activate venv if not already active
    if [[ -z "$VIRTUAL_ENV" ]] && [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        echo -e "${_NEEDLE_BLUE}Activating virtual environment...${_NEEDLE_NC}"
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
fi

# Check if LAW package is available (only fail for local, warn for remote)
if ! command -v law &> /dev/null; then
    if [[ -n "$LAW_HTCONDOR_JOB_NUMBER" ]] || [[ -n "$LAW_JOB_INIT_DIR" ]]; then
        echo -e "${_NEEDLE_ORANGE}Warning: LAW not found, but continuing for remote execution${_NEEDLE_NC}"
    else
        echo -e "${_NEEDLE_ORANGE}The package LAW could not be found, is your virtual environment active?${_NEEDLE_NC}"
        return 1
    fi
fi

# Check if the script was already sourced
if [[ -n "$NEEDLE_ENV_ACTIVE" ]]; then
    echo -e "${_NEEDLE_GREEN}$ENV_NAME environment is already active.${_NEEDLE_NC}"
    return 0
fi

# save current values so deactivate can restore them
export _OLD_PYTHONPATH="$PYTHONPATH"
export _OLD_PS1="$PS1"
# use resolved script dir so LAW_HOME is always correct
export LAW_HOME="$SCRIPT_DIR"
export LAW_CONFIG_FILE="$LAW_HOME/law.cfg"
export NEEDLE_ENV_ACTIVE=1

# Export FAIR_UNIVERSE_DATA path for dataset access
export FAIR_UNIVERSE_DATA="/data/dust/group/atlas/needle/FAIRUnv/UncertaintyChallenge_2024/ProcessedData_v1_2025-10-03/CombData-part0.parquet"

# use absolute paths so PYTHONPATH works regardless of cwd
for p in "preprocessor" "ml" "."; do
    abs_p="$LAW_HOME/$p"
    if [[ ":$PYTHONPATH:" != *":$abs_p:"* ]]; then
        export PYTHONPATH="$abs_p:$PYTHONPATH"
    fi
done

# load shell completions for both bash and zsh
# TODO: add for fish shell types
if [[ $- == *i* ]]; then
    if [[ -n "$BASH_VERSION" ]]; then
        . "$(law completion)" 2>/dev/null || true
    elif [[ -n "$ZSH_VERSION" ]]; then
        eval "$(law completion --zsh 2>/dev/null)" || true
    fi
fi

export PS1="($ENV_NAME):$PS1"

# fully restore env: unset all exported vars, not just PYTHONPATH/PS1
deactivate() {
    export PYTHONPATH="$_OLD_PYTHONPATH"
    export PS1="$_OLD_PS1"
    unset NEEDLE_ENV_ACTIVE
    unset LAW_HOME
    unset LAW_CONFIG_FILE
    unset FAIR_UNIVERSE_DATA
    unset _OLD_PYTHONPATH
    unset _OLD_PS1
    unset -f deactivate
    # guard: only unalias if the alias exists!
    alias exit &>/dev/null && unalias exit
    echo -e "\033[0;32mExited $ENV_NAME environment\033[0m"
    return 0
}

alias exit="deactivate"

# clean up colour vars so they don't pollute the users env
unset _NEEDLE_RED _NEEDLE_GREEN _NEEDLE_ORANGE _NEEDLE_NC _NEEDLE_BLUE

echo -e "\033[0;32mActivated $ENV_NAME environment. (Exit with 'exit' or 'deactivate')\033[0m"
