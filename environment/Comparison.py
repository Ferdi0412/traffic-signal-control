"""
Compare and generates graphs for trained traffic light agent vs a SCATS traffic model based on simulation time of 2500 seconds

To run code:
python environment/Comparison.py
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
from scats import SCATS
import argparse
import wandb
import os
import matplotlib.pyplot as plt
from Plotter import Plotter

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
    agent.load("./best_model.pth")

    # Set to true to generate GIF for both DQN and SCATS simulation
    render = True

    # Run a single episode using DQN agent
    agent.run_without_training(render)

    # Get metrics from DQN for comparison
    DQNcompare_reward, DQNcompare_deltaq, DQNcompare_longwait = agent._get_compare_rewards()
    DQNavg_qlength, DQNtotal_waittime, DQNthroughput  = agent._get_comparison_metrics()

    # Initiate SCATS model
    sumo = SumoInterface(**sumo_config)
    SCATS_traffic = SCATS(sumo)

    # Run a single episode using SCATS
    SCATS_traffic.single_epoch_run(unified_config['gym']['max_simtime'], render, True)

    # Get metrics from SCATS for comparison
    SCATScompare_reward, SCATScompare_deltaq, SCATScompare_longwait = SCATS_traffic._get_compare_rewards()
    SCATSavg_qlength, SCATStotal_waittime, SCATSthroughput = SCATS_traffic._get_comparison_metrics()

    # Dictionaries for plotting of bar charts
    directions_12x1 = {0: 'North Lane 1', 1: 'North Lane 2', 2: 'North Lane 3', 3: 'East Lane 1',
                  4: 'East Lane 2', 5: 'East Lane 3', 6: 'South Lane 1', 7: 'South Lane 2',
                  8: 'South Lane 3', 9: 'West Lane 1', 10: 'West Lane 2', 11: 'West Lane 3'}

    directions_4x1 = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}

    data_dict_rewards = {
                          'scenarios': ['DQNAgent', 'SCATS'],
                          'plt_rewards': [DQNcompare_reward, SCATScompare_reward],
                          'plt_deltaq': [DQNcompare_deltaq, SCATScompare_deltaq],
                          'plt_longwait': [DQNcompare_longwait, SCATScompare_longwait],
                          'xlabel': 'DQN vs SCATS',
                          'ylabel': 'Reward Components',
                          'title': 'Rewards Comparison Between DQN and SCATS',
                          'save_dir': True
                      }

    data_dict_qlength = {
            'data1': DQNavg_qlength,
            'data2': SCATSavg_qlength,
            'barlabel': [f"{directions_12x1[i]}" for i in range(len(DQNavg_qlength))],
            'label1': "DQN",
            'label2': "SCATS",
            'xlabel': 'NSEW',
            'ylabel': 'Queue Length',
            'title': 'Queue Length Comparison Between DQN and SCATS',
            'save_dir': True
        }

    data_dict_waittime = {
            'data1': DQNtotal_waittime,
            'data2': SCATStotal_waittime,
            'barlabel': [f"{directions_12x1[i]}" for i in range(len(DQNtotal_waittime))],
            'label1': "DQN",
            'label2': "SCATS",
            'xlabel': 'NSEW',
            'ylabel': 'Wait Time (s)',
            'title': 'Wait Time Comparison Between DQN and SCATS',
            'save_dir': True
        }

    data_dict_throughput = {
            'data1': DQNthroughput,
            'data2': SCATSthroughput,
            'barlabel': [f"{directions_4x1[i]}" for i in range(len(DQNthroughput))],
            'label1': "DQN",
            'label2': "SCATS",
            'xlabel': 'NSEW',
            'ylabel': 'Throughput (Cars/sec)',
            'title': 'Throughput Comparison Between DQN and SCATS',
            'save_dir': True
        }

    # Initialize plotter class to plot bar charts
    plotter = Plotter(figsize=(14, 6))

    # Plot comparisons between DQN and SCATS model
    plotter.plot_rewards_comparison(data_dict_rewards)
    plotter.plot_metrics_comparison(data_dict_qlength)
    plotter.plot_metrics_comparison(data_dict_waittime)
    plotter.plot_metrics_comparison(data_dict_throughput)



    