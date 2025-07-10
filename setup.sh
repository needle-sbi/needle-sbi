export LAW_HOME=$(pwd)
export LAW_CONFIG_FILE="$LAW_HOME/law.cfg"
export PYTHONPATH="./preprocessor/:ml:.:$PYTHONPATH"  # blame it on LAW that we have to overwrite PYTHONPATH
source "$(law completion)"
echo -e "\e[32mNEEDLE Setup successful!\e[0m"
