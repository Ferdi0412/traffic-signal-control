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

## Note

Due to SUMO (the actual simulator, not our code) implementation for **traci** which I believe uses C++ threads, you will likely have at the end of the code a message similar to the following:

"<span style="color: red">**SumoInterface.\_\_del\_\_**</span> error('required argument is not an integer')"

"Error: tcpip::Socket::recvAndCheck @ recv: ..."

This is **not an issue**, just a quirk of the library we use. It will in no way impact the training, as this ONLY EVER occurs when python is cleaning up the modules.

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