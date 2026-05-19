#!/bin/bash
# Activate the NEEDLE environment variables for this project.
# Must be sourced, not executed:  source setup.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]] 2>/dev/null; then
    echo "Error: This script must be sourced. Run: source setup.sh"
    exit 1
fi
if [[ -n "$ZSH_EVAL_CONTEXT" ]] && [[ "$ZSH_EVAL_CONTEXT" != *:file* ]]; then
    echo "Error: This script must be sourced. Run: source setup.sh"
    return 1
fi

_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"

export LAW_HOME="$_SETUP_DIR/.law"
export LAW_CONFIG_FILE="$_SETUP_DIR/law.cfg"
export SCRIPT_DIR="$_SETUP_DIR"

# Injected by `needle init`: parent directory of the installed law_tasks package.
# Ensures `law index` can find NEEDLE's built-in tasks regardless of working directory.
_NEEDLE_LAW_TASKS_PARENT="{{NEEDLE_LAW_TASKS_PARENT}}"
if [[ -n "$_NEEDLE_LAW_TASKS_PARENT" ]]; then
    export PYTHONPATH="$_NEEDLE_LAW_TASKS_PARENT${PYTHONPATH:+:$PYTHONPATH}"
fi
unset _NEEDLE_LAW_TASKS_PARENT

unset _SETUP_DIR

if [[ $- == *i* ]]; then
    if [[ -n "$BASH_VERSION" ]]; then
        . "$(law completion)" 2>/dev/null || true
    elif [[ -n "$ZSH_VERSION" ]]; then
        eval "$(law completion --zsh 2>/dev/null)" || true
    fi
fi

echo "NEEDLE environment activated."
echo "  LAW_HOME=$LAW_HOME"
echo "  LAW_CONFIG_FILE=$LAW_CONFIG_FILE"
