#!/bin/bash
export LAW_HOME=$(pwd)
export LAW_CONFIG_FILE="$LAW_HOME/law.cfg"
export PYTHONPATH="./preprocessor/:ml:.:$PYTHONPATH"  # blame it on LAW that we have to overwrite PYTHONPATH
# Only source completion if we're in an interactive bash shell
if [[ $- == *i* ]] && [[ -n "$BASH_VERSION" ]]; then
    . "$(law completion)" 2>/dev/null || true
fi
echo -e "\e[32mNEEDLE Setup successful!\e[0m"
