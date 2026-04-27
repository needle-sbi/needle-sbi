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

# Check if LAW package is available
if ! command -v law &> /dev/null; then
    echo -e "${_NEEDLE_ORANGE}The package LAW could not be found, is your virtual environment active?${_NEEDLE_NC}"
    return 1
fi

# Check if the script was already sourced
if [[ -n "$NEEDLE_ENV_ACTIVE" ]]; then
    echo -e "${_NEEDLE_GREEN}$ENV_NAME environment is already active.${_NEEDLE_NC}"
    return 0
fi

# resolve script dir instead of relying on pwd (bash/zsh compatible)
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
# save current values so deactivate can restore them
export _OLD_PYTHONPATH="$PYTHONPATH"
export _OLD_PS1="$PS1"
# use resolved script dir so LAW_HOME is always correct
export LAW_HOME="$SCRIPT_DIR/.law"
export LAW_CONFIG_FILE="$LAW_HOME/law.cfg"

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
_deactivate() {
    export PYTHONPATH="$_OLD_PYTHONPATH"
    export PS1="$_OLD_PS1"
    export NEEDLE_ENV_ACTIVE=0
    unset NEEDLE_ENV_ACTIVE
    unset SCRIPT_DIR
    unset LAW_HOME
    unset LAW_CONFIG_FILE
    unset _OLD_PYTHONPATH
    unset _OLD_PS1
    unset -f deactivate
    # guard: only unalias if the alias exists!
    alias exit &>/dev/null && unalias exit
    echo -e "\033[0;32mExited $ENV_NAME environment\033[0m"
    return 0
}

alias exit="_deactivate"
alias deactivate="_deactivate"

# Backup in case .venv was exited before needle env
alias needle_deactivate="_deactivate"
alias needle_exit="_deactivate"

bash "tui/plain_text/welcome_message.sh"

echo -e "${_NEEDLE_GREEN}Activated the $ENV_NAME environment. (Exit with 'exit', 'deactivate' or 'needle_exit').${_NEEDLE_NC}"
echo -e "${_NEEDLE_GREEN}For a full reset in the same shell, use 'unset NEEDLE_ENV_ACTIVE' then re-source this script.${_NEEDLE_NC}"

# clean up colour vars so they don't pollute the users env
unset _NEEDLE_RED _NEEDLE_GREEN _NEEDLE_ORANGE _NEEDLE_NC _NEEDLE_BLUE

export NEEDLE_ENV_ACTIVE=1
