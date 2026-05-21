#!/usr/bin/env bash
set -euo pipefail

# extract the python executable
extract_python_exe() {
    # prefer active env python; fallback to system
    if command -v python3 >/dev/null 2>&1; then
        echo "$(command -v python3)"
    else
        echo "$(command -v python)"
    fi
}

# extract the pip command
extract_pip_command() {
    # always use "python -m pip" to avoid calling a wrong/old pip binary
    local python_exe
    python_exe="$(extract_python_exe)"
    echo "${python_exe} -m pip install"
}

# check if input directory is a python extension and install the module
install_isaaclab_extension() {
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    pip_command=$(extract_pip_command)

    # if the directory contains setup.py then install the python module
    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        # NOTE: --editable is an option to "pip install", not to the bare "pip" command
        $pip_command -e "$1"
    fi
}

echo "Installing pip packages on source directory..."
export -f extract_python_exe
export -f extract_pip_command
export -f install_isaaclab_extension

find -L "./source" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_isaaclab_extension "{}"' \;
