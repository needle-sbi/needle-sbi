#!/bin/bash
# Activate the NEEDLE environment variables for this project.
# Must be sourced, not executed:  source setup.sh

ENV_NAME="NEEDLE"

# Colors — cleaned up at the end to avoid polluting the user's env
_NEEDLE_RED='\033[0;31m'
_NEEDLE_GREEN='\033[0;32m'
_NEEDLE_ORANGE='\033[0;33m'
_NEEDLE_NC='\033[0m'

# Guard: source works, direct execution does not
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] 2>/dev/null; then
    echo -e "\033[0;31mError: This script must be sourced. Run: source setup.sh\033[0m"
    exit 1
fi
if [[ -n "$ZSH_EVAL_CONTEXT" ]] && [[ "$ZSH_EVAL_CONTEXT" != *:file* ]]; then
    echo -e "\033[0;31mError: This script must be sourced. Run: source setup.sh\033[0m"
    return 1
fi

# Check if LAW is available (only hard-fail for local sessions)
if ! command -v law &> /dev/null; then
    if [[ -n "$LAW_HTCONDOR_JOB_NUMBER" ]] || [[ -n "$LAW_JOB_INIT_DIR" ]]; then
        echo -e "${_NEEDLE_ORANGE}Warning: LAW not found, but continuing for remote execution${_NEEDLE_NC}"
    else
        echo -e "${_NEEDLE_ORANGE}LAW not found — is your virtual environment active?${_NEEDLE_NC}"
        unset _NEEDLE_RED _NEEDLE_GREEN _NEEDLE_ORANGE _NEEDLE_NC
        return 1
    fi
fi

# Guard: already sourced
if [[ -n "$NEEDLE_ENV_ACTIVE" ]]; then
    echo -e "${_NEEDLE_GREEN}$ENV_NAME environment is already active.${_NEEDLE_NC}"
    unset _NEEDLE_RED _NEEDLE_GREEN _NEEDLE_ORANGE _NEEDLE_NC
    return 0
fi

# Resolve script directory (bash/zsh compatible)
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"

# Save current env so _deactivate can restore it
export _OLD_PYTHONPATH="${PYTHONPATH:-}"
export _OLD_PS1="$PS1"

export LAW_HOME="$SCRIPT_DIR"
export LAW_CONFIG_FILE="$SCRIPT_DIR/law.cfg"

# Load shell completions
if [[ $- == *i* ]]; then
    if [[ -n "$BASH_VERSION" ]]; then
        . "$(law completion)" 2>/dev/null || true
    elif [[ -n "$ZSH_VERSION" ]]; then
        eval "$(law completion --zsh 2>/dev/null)" || true
    fi
fi

export PS1="($ENV_NAME) $PS1"

_deactivate() {
    if [[ -n "$_OLD_PYTHONPATH" ]]; then
        export PYTHONPATH="$_OLD_PYTHONPATH"
    else
        unset PYTHONPATH
    fi
    export PS1="$_OLD_PS1"
    unset NEEDLE_ENV_ACTIVE SCRIPT_DIR LAW_HOME LAW_CONFIG_FILE _OLD_PYTHONPATH _OLD_PS1
    unset -f _deactivate
    for _a in exit deactivate needle_deactivate needle_exit; do
        alias "$_a" &>/dev/null && unalias "$_a"
    done
    unset _a
    echo -e "\033[0;32mExited $ENV_NAME environment\033[0m"
    return 0
}

alias exit="_deactivate"
alias deactivate="_deactivate"
alias needle_deactivate="_deactivate"
alias needle_exit="_deactivate"

_NEEDLE_TUI_DIR=$(python3 -c "import needle, pathlib; print(pathlib.Path(next(iter(needle.__path__))) / 'tui')" 2>/dev/null)
if [[ -n "$_NEEDLE_TUI_DIR" && -f "$_NEEDLE_TUI_DIR/plain_text/welcome_message.sh" ]]; then
    (NEEDLE_TUI_DIR="$_NEEDLE_TUI_DIR" bash "$_NEEDLE_TUI_DIR/plain_text/welcome_message.sh")
fi
unset _NEEDLE_TUI_DIR

echo -e "${_NEEDLE_GREEN}Activated the $ENV_NAME environment. (Exit with 'exit', 'deactivate' or 'needle_exit').${_NEEDLE_NC}"
echo -e "${_NEEDLE_GREEN}For a full reset in the same shell, use 'unset NEEDLE_ENV_ACTIVE' then re-source this script.${_NEEDLE_NC}"

# Clean up colour vars so they don't pollute the user's env
unset _NEEDLE_RED _NEEDLE_GREEN _NEEDLE_ORANGE _NEEDLE_NC

export NEEDLE_ENV_ACTIVE=1
