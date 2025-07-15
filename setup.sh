#!/bin/bash

ENV_NAME="NEEDLE"

# Check if LAW package is available
if ! command -v law &> /dev/null; then
    echo -e "\033[0;33mThe package LAW not found, is your virtual environment active?\033[0m"
    return 1
fi

# Check if the script was already sourced
if [[ -n "$NEEDLE_ENV_ACTIVE" ]]; then
    echo -e "\033[0;32m$ENV_NAME environment is already active.\033[0;32m"
    return 0
fi

# Save old shell variables and export the new ones
export _OLD_PYTHONPATH="$PYTHONPATH"
export _OLD_PS1="$PS1"
export LAW_HOME=$(pwd)
export LAW_CONFIG_FILE="$LAW_HOME/law.cfg"
export NEEDLE_ENV_ACTIVE=1

# Blame it on LAW that we have to overwrite PYTHONPATH
for p in "preprocessor" "ml" "."; do
    if [[ ":$PYTHONPATH:" != *":$p:"* ]]; then
        export PYTHONPATH="$p:$PYTHONPATH"
    fi
done

# Only source LAW completion if we're in an interactive bash shell
if [[ $- == *i* ]] && [[ -n "$BASH_VERSION" ]]; then
    . "$(law completion)" 2>/dev/null || true
fi

export PYTHONPATH
export PS1="($ENV_NAME):$PS1"

deactivate() {
    export PYTHONPATH="$_OLD_PYTHONPATH"
    export PS1="$_OLD_PS1"
    unset NEEDLE_ENV_ACTIVE
    unset -f deactivate
    unalias exit
    echo -e "Exited $ENV_NAME environment"
    return 0
}

alias exit="deactivate"