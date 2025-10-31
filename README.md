# Traffic Signal Control

This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors

- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

# Setup
## Setting up SUMO and Python

The only new dependency from previous iteration was **PyTorch**. To reset the environnment, you can use any of the following steps:
1) Option 1: Remove old conda environment and install new
    - `conda remove --name traffic --all`
    - `chmod +x setup.sh && ./setup.sh`
2) Option 2: Use the **reset** script
    - `chmod +x reset.sh && ./reset.sh`

Next, activate your conda environment

```sh
conda activate traffic
```

## Running the Code

Run the NN code:

```sh
python project/NeuralNetwork.py
```

## File Structure

```txt
/
├── project/                   # This is where we keep all our main code at the moment
│   │                          # "project/" was previously "environment/
│   ├── import_sumo.py         # traci import issue workaround
│   ├── sumo_interface.py      # SumoInterface definition
│   ├── traffilightgymsumo.py  # Gym code
│   ├── NeuralNetwork.py       # NN implementation 
├── sumo_networks/             # This is where the SUMO config files are
```