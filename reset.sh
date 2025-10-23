#!/bin/bash
set -e # Exit if any command fails

### This file clears the Conda environment
### and re-runs setup.sh

# Pretty formatting
DIM="\033[2m"
BLU="\033[1m\033[34m"
YEL="\033[1m\033[33m"
RED="\033[1m\033[31m"
END="\033[0m"
PRE="${DIM}>> ${END}"

# Extract the econda environment name from the environment.yml file
CONDA_ENV_NAME=$(grep '^name:' environment.yml | awk '{print $2}')
if [ -z "$CONDA_ENV_NAME" ]; then
	CONDA_ENV_NAME="traffic"
	echo
    echo -e "${PRE}${YEL}Warning:${END} Failed to extract name from environment.yml, verify name there is ${BLU}${CONDA_ENV_NAME}${END}"
fi

# Conda for bash shell
eval "$(conda shell.bash hook)"

# Deactivate conda environment if currently activated
if [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV_NAME}" ]; then
    conda deactivate
fi

# If the environment exists, remove it
if conda env list | grep -q "^${CONDA_ENV_NAME}"; then
    echo
    echo -e "${PRE}${YEL}Removing existing ${BLU}${CONDA_ENV_NAME}${END}"
    conda remove -n "${CONDA_ENV_NAME}" --all -y
fi

./setup.sh