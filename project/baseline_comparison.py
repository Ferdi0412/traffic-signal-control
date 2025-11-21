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
from gym import TrafficGym
from learning_agent import DQNAgent
from giffer import SumoGif
from scats import SCATS
import argparse
import wandb
import os
import matplotlib.pyplot as plt
from baseline_plot import Plotter

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

    random_seed = np.random.randint(100)

    unified_config = {
        # SUMO Config
        'sumo': {
            "fname": args.file,
            "gui": args.gui,
            "seed": random_seed,
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

    # Set to true to generate GIF for both DQN and SCATS simulation
    render = True

    # Run a single episode using DQN agent
    # agent.run_without_training(render)

    # episodes to run
    episodes = 100

    # Generate a gif at one random episode
    render_episode = np.random.randint(episodes)

    #initialise comparison variables for DQN
    DQNcompare_reward = np.zeros(0, dtype=float)
    DQNcompare_deltaq = np.zeros(0, dtype=float)
    DQNcompare_longwait = np.zeros(0, dtype=float)
    DQNavg_qlength = np.zeros((12,1), dtype=float)
    DQNtotal_waittime = np.zeros((12,1), dtype=float)
    DQNthroughput = np.zeros((4,1), dtype=float)

    # run X episodes with random seed for DQN
    for i in range (episodes):
        if i == render_episode:
            render = True
        else:
            render = False
        agent.run_without_training(render)
        _DQNcompare_reward, _DQNcompare_deltaq, _DQNcompare_longwait = agent._get_compare_rewards()
        _DQNavg_qlength, _DQNtotal_waittime, _DQNthroughput  = agent._get_comparison_metrics()
        if i == 0:
            DQNcompare_reward = _DQNcompare_reward
            DQNcompare_deltaq = _DQNcompare_deltaq
            DQNcompare_longwait = _DQNcompare_longwait
            DQNavg_qlength = _DQNavg_qlength.reshape(-1, 1)
            DQNtotal_waittime = _DQNtotal_waittime.reshape(-1, 1)
            DQNthroughput = _DQNthroughput.reshape(-1, 1)
        else:
            DQNcompare_reward = np.hstack((DQNcompare_reward, _DQNcompare_reward))
            DQNcompare_deltaq = np.hstack((DQNcompare_deltaq, _DQNcompare_deltaq))
            DQNcompare_longwait = np.hstack((DQNcompare_longwait, _DQNcompare_longwait))
            DQNavg_qlength = np.hstack((DQNavg_qlength, _DQNavg_qlength.reshape(-1, 1)))
            DQNtotal_waittime = np.hstack((DQNtotal_waittime, _DQNtotal_waittime.reshape(-1, 1)))
            DQNthroughput = np.hstack((DQNthroughput, _DQNthroughput.reshape(-1, 1)))
        agent._random()
        agent.env.reset()
        print("DQN Episode " + str(i) + " completed")

    # average values over episodes ran for DQN
    mean_DQNcompare_reward = DQNcompare_reward.mean()
    std_DQNcompare_reward = DQNcompare_reward.std()
    mean_DQNcompare_deltaq = DQNcompare_deltaq.mean()
    std_DQNcompare_deltaq = DQNcompare_deltaq.std()
    mean_DQNcompare_longwait = DQNcompare_longwait.mean()
    std_DQNcompare_longwait = DQNcompare_longwait.std()
    mean_DQNavg_qlength = np.mean(DQNavg_qlength, axis=1)
    std_DQNavg_qlength = np.std(DQNavg_qlength, axis=1)
    mean_DQNtotal_waittime = np.mean(DQNtotal_waittime, axis=1)
    std_DQNtotal_waittime = np.std(DQNtotal_waittime, axis=1)
    mean_DQNthroughput = np.mean(DQNthroughput, axis=1)
    std_DQNthroughput = np.std(DQNthroughput, axis=1)

    # Get metrics from DQN for comparison
    # DQNcompare_reward, DQNcompare_deltaq, DQNcompare_longwait = agent._get_compare_rewards()
    # DQNavg_qlength, DQNtotal_waittime, DQNthroughput  = agent._get_comparison_metrics()

    # Initiate SCATS model
    sumo = SumoInterface(**sumo_config)
    SCATS_traffic = SCATS(sumo)

    # Run a single episode using SCATS
    # SCATS_traffic.single_epoch_run(unified_config['gym']['max_simtime'], render, True)

    # initialise comparison variables for SCATS
    SCATScompare_reward = np.zeros(0, dtype=float)
    SCATScompare_deltaq = np.zeros(0, dtype=float)
    SCATScompare_longwait = np.zeros(0, dtype=float)
    SCATSavg_qlength = np.zeros((12,1), dtype=float)
    SCATStotal_waittime = np.zeros((12,1), dtype=float)
    SCATSthroughput = np.zeros((4,1), dtype=float)

    # run 100 episodes with random seed for SCATS
    for i in range (episodes):
        if i == render_episode:
            render = True
        else:
            render = False
        SCATS_traffic.single_epoch_run(unified_config['gym']['max_simtime'], render, True)
        _SCATScompare_reward, _SCATScompare_deltaq, _SCATScompare_longwait = SCATS_traffic._get_compare_rewards()
        _SCATSavg_qlength, _SCATStotal_waittime, _SCATSthroughput  = SCATS_traffic._get_comparison_metrics()
        if i == 0:
            SCATScompare_reward = _SCATScompare_reward
            SCATScompare_deltaq = _SCATScompare_deltaq
            SCATScompare_longwait = _SCATScompare_longwait
            SCATSavg_qlength = _SCATSavg_qlength.reshape(-1, 1)
            SCATStotal_waittime = _SCATStotal_waittime.reshape(-1, 1)
            SCATSthroughput = _SCATSthroughput.reshape(-1, 1)
        else:
            SCATScompare_reward = np.hstack((SCATScompare_reward, _SCATScompare_reward))
            SCATScompare_deltaq = np.hstack((SCATScompare_deltaq, _SCATScompare_deltaq))
            SCATScompare_longwait = np.hstack((SCATScompare_longwait, _SCATScompare_longwait))
            SCATSavg_qlength = np.hstack((SCATSavg_qlength, _SCATSavg_qlength.reshape(-1, 1)))
            SCATStotal_waittime = np.hstack((SCATStotal_waittime, _SCATStotal_waittime.reshape(-1, 1)))
            SCATSthroughput = np.hstack((SCATSthroughput, _SCATSthroughput.reshape(-1, 1)))
        SCATS_traffic._random()
        SCATS_traffic.reset()
        print("SCATS Episode " + str(i) + " completed")

    # average values over episodes ran for SCATS
    mean_SCATScompare_reward = SCATScompare_reward.mean()
    std_SCATScompare_reward = SCATScompare_reward.std()
    mean_SCATScompare_deltaq = SCATScompare_deltaq.mean()
    std_SCATScompare_deltaq = SCATScompare_deltaq.std()
    mean_SCATScompare_longwait = SCATScompare_longwait.mean()
    std_SCATScompare_longwait = SCATScompare_longwait.std()
    mean_SCATSavg_qlength = np.mean(SCATSavg_qlength, axis=1)
    std_SCATSavg_qlength = np.std(SCATSavg_qlength, axis=1)
    mean_SCATStotal_waittime = np.mean(SCATStotal_waittime, axis=1)
    std_SCATStotal_waittime = np.std(SCATStotal_waittime, axis=1)
    mean_SCATSthroughput = np.mean(SCATSthroughput, axis=1)
    std_SCATSthroughput = np.std(SCATSthroughput, axis=1)

    # Get metrics from SCATS for comparison
    # SCATScompare_reward, SCATScompare_deltaq, SCATScompare_longwait = SCATS_traffic._get_compare_rewards()
    # SCATSavg_qlength, SCATStotal_waittime, SCATSthroughput = SCATS_traffic._get_comparison_metrics()

    # Dictionaries for plotting of bar charts
    directions_12x1 = {0: 'North Lane 1', 1: 'North Lane 2', 2: 'North Lane 3', 3: 'East Lane 1',
                  4: 'East Lane 2', 5: 'East Lane 3', 6: 'South Lane 1', 7: 'South Lane 2',
                  8: 'South Lane 3', 9: 'West Lane 1', 10: 'West Lane 2', 11: 'West Lane 3'}

    directions_4x1 = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}

    data_dict_rewards = {
                          'scenarios': ['DQNAgent', 'SCATS'],
                          'plt_rewards': [mean_DQNcompare_reward, mean_SCATScompare_reward],
                          'plt_deltaq': [mean_DQNcompare_deltaq, mean_SCATScompare_deltaq],
                          'plt_longwait': [mean_DQNcompare_longwait, mean_SCATScompare_longwait],
                          'plt_rewards_dev': [std_DQNcompare_reward, std_SCATScompare_reward],
                          'plt_deltaq_dev': [std_DQNcompare_deltaq, std_SCATScompare_deltaq],
                          'plt_longwait_dev': [std_DQNcompare_longwait, std_SCATScompare_longwait],
                          'xlabel': 'DQN vs SCATS',
                          'ylabel': 'Reward Components',
                          'title': 'Rewards Comparison Between DQN and SCATS',
                          'save_dir': True
                      }

    data_dict_qlength = {
            'data1': mean_DQNavg_qlength,
            'data2': mean_SCATSavg_qlength,
            'dev1': std_DQNavg_qlength,
            'dev2': std_SCATSavg_qlength,
            'barlabel': [f"{directions_12x1[i]}" for i in range(len(mean_DQNavg_qlength))],
            'label1': "DQN",
            'label2': "SCATS",
            'xlabel': 'NSEW',
            'ylabel': 'Queue Length',
            'title': 'Queue Length Comparison Between DQN and SCATS',
            'save_dir': True
        }

    data_dict_waittime = {
            'data1': mean_DQNtotal_waittime,
            'data2': mean_SCATStotal_waittime,
            'dev1': std_DQNtotal_waittime,
            'dev2': std_SCATStotal_waittime,
            'barlabel': [f"{directions_12x1[i]}" for i in range(len(mean_DQNtotal_waittime))],
            'label1': "DQN",
            'label2': "SCATS",
            'xlabel': 'NSEW',
            'ylabel': 'Wait Time (s)',
            'title': 'Wait Time Comparison Between DQN and SCATS',
            'save_dir': True
        }

    data_dict_throughput = {
            'data1': mean_DQNthroughput,
            'data2': mean_SCATSthroughput,
            'dev1': std_DQNthroughput,
            'dev2': std_SCATSthroughput,
            'barlabel': [f"{directions_4x1[i]}" for i in range(len(mean_DQNthroughput))],
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



    