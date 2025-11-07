# Traffic Signal Control

This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors

- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

# Setup
## Setting up SUMO and Python
Our environment is named `traffic`:
```sh
conda activate traffic
```

> No new dependencies since submission 2
> If you don't have the previous environment for this project (`traffic`): `chmod +x reset.sh && ./reset.sh`


## Running the Code
Run the learning agent code:

```sh
python project/LearningAgent.py
```

## File Structure

```txt
/
├── project/                   # This is where we keep all our main code at the moment
│   │
│   ├── import_sumo.py         # traci import issue workaround
│   ├── sumo_interface.py      # SumoInterface definition
│   ├── traffilightgymsumo.py  # Gym code
│   ├── LearningAgent.py       # NN and RL Agent implementation 
├── sumo_networks/             # This is where the SUMO config files are, as needed by the sim
```