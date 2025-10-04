#!/bin/bash
set -e # Exit if any command fails

CONDA_ENV_NAME="traffic"
PYTHON_VERSION=3.6.10

# Conda for bash shell
eval "$(conda shell.bash hook)"

# If environment does not exist, create it
if ! conda env list | grep -q "^${CONDA_ENV_NAME}"; then
	echo "=== Creating Conda Environment ==="
	conda create -n "$CONDA_ENV_NAME" python=$PYTHON_VERSION
fi

# Install openai gym requirements
echo "=== Installing requirements using PIP ==="
conda activate traffic
conda install pip
pip install -r requirements.txt

# Install SUMO
echo "=== Installing SUMO ==="
sudo apt-get install sumo sumo-tools sumo-doc
if ! grep -q "SUMO_HOME" ~/.bashrc; then
	echo "=== Adding SUMO environment variable ==="
	echo "" >> ~/.bashrc
	echo "# SUMO Environment Variable" >> ~/.bashrc
	echo "export SUMO_HOME=\"/usr/share/sumo\"" >> ~/.bashrc
fi
export SUMO_HOME="/usr/share/sumo"