# Traffic Signal Control

This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

> Our code outputs GIFs that will be used as our video, see the bottom of this file, or look in the `video/` directory.

**Authors:**
- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

> NOTE: Due to the implementation of the SUMO library, you may see a message "Error: tcpip::Socket::recvAndCheck @ recv..." - this is normal, and due to SUMO's threading model for the library

## Setup
> Since last submission, the following `pip` libraries have been added: `wandb`, `matplotlib`

> Note also that we require **PyTorch**, however as the version needed might change depending on your system, so it has been left without a version - if you need a specific version, input this to `requirements.txt`

```sh
## OPTION 1) If you have conda environment `traffic`
conda activate traffic
pip install -r requirements.txt

## OPTION 2) If you don't have conda environment, or last step failed, use one of the following
# A)
sudo apt-get install sumo sumo-tools sumo-doc
conda env create -f environment.yml

# OR #

# B)
./reset.sh
```

## Validation Code
This runs a quick episode of out trained model and outputs a gif of it's performance.
```sh
conda activate traffic # If not already active

# From the root of this project
python project/validation.py
```

> This will print traffic performance metrics from a single episode ran using the DQN trained model, and output a gif showing the performance, look out for the *green text* `Saved gif to ./DQN_...` print to see the resulting filename.

## Training Code
This trains the model, which takes a very long time (several hours), and saves a checkpoint every `100` episodes.
```sh
conda activate traffic # If not already active

# From the root of this project
python project/learning_agent.py
```

As we are using `wandb`, this will ask for where to store the wandb data. **Enter value** `3` when this prompt comes up:
```txt
wandb: Enter your choice:
```

## Project Overview
```txt
root /
├── project/
│    │   ## For your usage ##
│    ├── learning_agent.py # Train
│    ├── validation.py     # Test and output gif of performance
│    │
│    │   ## For internal usage ##
│    ├── baseline_comparison.py # Compare scats and DQN
│    ├── baseline_plot.py       # Used by baseline_comparison.py
│    ├── giffer.py              # SumoGif class
│    ├── gym.py                 # TrafficGym class
│    ├── nn.py                  # NN and DuelingNN classes
│    ├── scats.py               # Scats implementation
│    ├── sumo_interface.py      # SumoInterface class
│    └── utils.py               # ReplayBuffer and reasonable_actions
│
├── checkpoints/
│    └── best_model.pth  # Best trained model
│
└── sumo-networks/
     └── ...              # SUMO configuration files
```

## Comparison
### DQN Video

![DQN Gif](videos/DQN_video.gif)


### Scats (baseline) Video
![Scats Gif](videos/SCATS_video.gif)