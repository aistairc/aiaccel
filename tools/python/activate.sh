SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$($SCRIPT_PATH/miniforge3/bin/conda shell.bash hook 2> /dev/null)"