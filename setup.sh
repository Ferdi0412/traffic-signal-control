#!/bin/bash
set -e # Exit if any command fails

### This file installs SUMO and installs the appropriate
### conda environment

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

# If environment does not exist, create it
if ! conda env list | grep -q "^${CONDA_ENV_NAME}"; then
	echo
	echo -e "${PRE}Creating new conda environment ${BLU}${CONDA_ENV_NAME}${END}"
	# conda create -n "$CONDA_ENV_NAME" python=$PYTHON_VERSION
	conda env create -f environment.yml
fi

# Install SUMO
# This will run every time
echo -e "${PRE}${BLU}Installing SUMO${END}"
sudo apt-get install sumo sumo-tools sumo-doc

# Environment variables did not work on previous example
# Add environment variable if not already done
# if ! grep -q "SUMO_HOME" ~/.bashrc; then
	# echo
	# echo -e "${PRE}Adding ${BLU}${SUMO_HOME}${END} environment variable"
	# echo "" >> ~/.bashrc
	# echo "# SUMO Environment Variable" >> ~/.bashrc
	# echo "export SUMO_HOME=\"/usr/share/sumo\"" >> ~/.bashrc
# fi
# export SUMO_HOME="/usr/share/sumo"