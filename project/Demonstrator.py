"""
Demonstrate running of a single episode of trained traffic light agent and
generates a GIF to show simulation results based on simulation time of 2500 seconds

To run code:
python environment/Demonstrator.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sumo_interface import SumoInterface
from trafficlightgymsumo_NN_wandb import TrafficGym
from NN_wandb import DQNAgent
from giffer import SumoGif
import argparse
import wandb
import os

reasonable_actions = [
#0,  # All Red (Transition)
3,  # North Left+Forward
4,  # North Right Only
7,  # North All
24,  # East Left+Forward
32,  # East Right Only
56,  # East All
192,  # South Left+Forward
195,  # North Left+Forward + South Left+Forward
196,  # North Right + South Left+Forward
199,  # North All + South Left+Forward
256,  # South Right Only
259,  # North Left+Forward + South Right
260,  # North Right + South Right
263,  # North All + South Right
448,  # South All
451,  # North Left+Forward + South All
452,  # North Right + South All
455,  # North All + South All
1536,  # West Left+Forward
1560,  # East Left+Forward + West Left+Forward
1568,  # East Right + West Left+Forward
1592,  # East All + West Left+Forward
2048,  # West Right Only
2072,  # East Left+Forward + West Right
2080,  # East Right + West Right
2104,  # East All + West Right
3584,  # West All
3608,  # East Left+Forward + West All
3616,  # East Right + West All
3640  # East All + West All
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="map_2", help="SUMO file to use")
    parser.add_argument("-g", "--gui", action="store_true", help="Whether to show GUI")
    parser.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    unified_config = {
        # SUMO Config
        'sumo': {
            "fname": args.file,
            "gui": args.gui,
            "seed": 8,
        },
        
        # Gym  Config
        'gym': {
            "max_simtime": 2500,
            "no_of_sensors": 5,
            "traffic_rate_upstream": "High",
            "traffic_rate_downstream": "Low",
            "reward_weights": [0.01, 0.03],
            "action_repeat": 5
        },
        
        # Agent Config
        'agent': {
            'learning_rate': 0.000001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995,
            'buffer_size': 100000,
            'batch_size': 128,
            'target_update_freq': 100,
            'grad_clip': 1.0,
            'min_action_timer': 1,
            'dqn_variant': 'double_dueling_dqn'  # 'dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn'
        },
        
        # Training Config
        'training': {
            'num_episodes': 1000,
            'eval_interval': 50,
            'eval_episodes': 5,
            'gif_interval': 100
        },
        
        # Wandb Config
        'wandb': {
            'entity': "kaiyi-lam-ml",
            'project': "traffic-signal-control",
            'enabled': False,
            'name': args.wandb_name
        }
    }

    sumo_config = unified_config['sumo']
    gym_config = unified_config['gym'] 

    # Create the Gym environment
    env = TrafficGym(sumo_config, config=gym_config)

    # Environment parameters
    state_size = len(env._observe_NN())
    action_size = len(reasonable_actions)

    # Create agent with simplified constructor and load trained model
    agent = DQNAgent(state_size, action_size, env, unified_config)
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent.load(os.path.join(proj_root, "checkpoints", "best_model.pth"))

    # Set to true to generate GIF
    render = True

    # Run a single episode using DQN agent
    agent.run_without_training(render)