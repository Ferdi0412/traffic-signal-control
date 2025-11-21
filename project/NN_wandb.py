import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from trafficlightgymsumo_NN_wandb import TrafficGym
from giffer import SumoGif
import argparse
import wandb
import os
from datetime import datetime


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

class NN(nn.Module):
    """Standard Deep Q-Network for traffic signal control"""
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()

        # Define network layers
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)

class DuelingNN(nn.Module):
    """Dueling Deep Q-Network for traffic signal control"""
    def __init__(self, state_size, action_size):
        super(DuelingNN, self).__init__()
        
        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        features = self.feature_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage using dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for traffic signal control"""
    def __init__(self, state_size, action_size, env, unified_config, eval_env=None):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.eval_env = eval_env
        self.unified_config = unified_config
        
        self.gym_config = unified_config['gym']
        self.agent_config = unified_config['agent']
        self.wandb_config = unified_config['wandb']
        self.training_config = unified_config['training']
        
        self.log_wandb = self.wandb_config['enabled']
        self.wandb_testname = self.wandb_config['name']

        self.dqn_variant = self.agent_config.get('dqn_variant', 'dqn')  
        self.use_double_dqn = 'double' in self.dqn_variant
        self.use_dueling = 'dueling' in self.dqn_variant
        
        if self.wandb_testname:
            self.save_dir = os.path.join("./training", self.wandb_testname)
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = "./"
        
        self.config = self.agent_config
        
        self.epsilon = self.config['epsilon']
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.target_update_freq = self.config['target_update_freq']
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        NetworkClass = DuelingNN if self.use_dueling else NN
        self.policy_net = NetworkClass(state_size, action_size).to(self.device)
        self.target_net = NetworkClass(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # Replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'])
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []

        if self.log_wandb:
            wandb.init(
                entity=self.wandb_config['entity'],
                project=self.wandb_config['project'],
                name=self.wandb_config['name'],
                config=self.unified_config
            )
            wandb.watch(self.policy_net, log='all', log_freq=100)

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                self.policy_net.eval()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                self.policy_net.train()
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""        
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the DQN using experience replay"""
        if len(self.memory) < max(5000, self.batch_size * 10):
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q = torch.clamp(current_q, -100, 100)
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            else:
                # DQN
                next_q = self.target_net(next_states).max(1)[0]
            
            next_q = torch.clamp(next_q, -100, 100)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            target_q = torch.clamp(target_q, -100, 100)

        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Find gradient norm
        # grad_norm = 0.
        # for policy in self.policy_net.parameters():
        #     if policy.grad is not None:
        #         grad_norm += policy.grad.data.norm(2).item() ** 2
        # grad_norm = grad_norm ** 0.5

        #Clip gradient
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config['grad_clip'])
        self.optimizer.step()
        # if self.training_step % 100 == 0:  # Update LR every 100 training steps
        #     self.scheduler.step()
            
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        if self.log_wandb:
            wandb.log({
                'loss': loss.item(), #decreasing
                # 'mean_q_value': current_q.mean().item(), #should increase
                # 'td_error': td_error, #decreasing but not too low
                # 'grad_norm': grad_norm, #stable in 0.1-10 range
                # 'buffer_size' : len(self.memory) #should fill up in 10-20ep
            })
        
        return loss.item()
    
    def end_episode(self):
        """Decay epsilon once per episode"""
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon *= self.config['epsilon_decay']
            
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        print(f"Model saved to {filepath}")

        if self.log_wandb:
            wandb.save(filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {filepath}")

    def evaluate(self, num_episodes=5):
        eval_rewards = []

        for _ in range(num_episodes):
            action = np.random.randint(0, self.action_size)
            episode_reward = 0.
            eval_reward_components = []
            state = self.eval_env._observe_NN()
            for _ in range(self.gym_config['max_simtime']):
                action_idx = self.select_action(state, training=False)
                action = reasonable_actions[action_idx]
                next_state, reward, done, _, reward_components, _ = self.eval_env.step(action)
                eval_reward_components.append(reward_components)
                episode_reward += reward
                state = next_state
                if done:
                    break
            eval_rewards.append(episode_reward)
            self.eval_env.reset()

        return np.mean(eval_rewards)

    def run(self, num_episodes, eval_interval=50, eval_episodes=5, gif_interval=100, training = True):
        
        # Print unified configuration for verification
        print("\n" + "="*60)
        print("UNIFIED CONFIGURATION")
        print("="*60)
        for section, params in self.unified_config.items():
            print(f"\n[{section.upper()}]")
            for key, value in params.items():
                print(f"  {key}: {value}")
        print("="*60 + "\n")
        
        for episode in range(num_episodes):
            prev_action = None   
            episode_reward = 0.
            action_changes = 0
            ep_reward_components = []
            actions_taken = []
            state = self.env._observe_NN()

            # Create GIF every gif_interval episodes
            create_gif = (episode + 1) % gif_interval == 0
            gif = None
            if create_gif:
                gif_filename = os.path.join(self.save_dir, f"{episode + 1}.gif")

            for _ in range(self.gym_config['max_simtime']): 
                action_idx = self.select_action(state, training)
                action = reasonable_actions[action_idx]
                next_state, reward, done, step_count, reward_components, step_metrics = self.env.step(action)
                ep_reward_components.append(reward_components)
                
                # Store step metrics for episode-level aggregation
                if not hasattr(self, 'episode_metrics'):
                    self.episode_metrics = []
                self.episode_metrics.append(step_metrics)
                
                # Update GIF frame if creating GIF
                if create_gif and gif is not None:
                    gif.update_buffer()
                
                if prev_action is not None and action != prev_action:
                    action_changes += 1
                actions_taken.append(action)
                self.store_transition(state, action_idx, reward, next_state, done)
                if training:
                    loss = self.train()
                prev_action = action
                episode_reward += reward
                state = next_state
                if done:
                    break
            
            # Save GIF if one was created
            if create_gif and gif is not None:
                gif.save()
                print(f"GIF saved: {gif_filename}")
                
            # Print top 5 most frequent actions
            if episode % 50 == 0:
                actions_array = np.array(actions_taken)
                unique_actions, counts = np.unique(actions_array, return_counts=True)
                # Sort by count (descending) and get top 5
                sorted_indices = np.argsort(counts)[::-1]
                top_5_indices = sorted_indices[:min(5, len(sorted_indices))]
                
                print("Top 5 most frequent actions:")
                for i, idx in enumerate(top_5_indices):
                    action_value = unique_actions[idx]
                    action_count = counts[idx]
                    percentage = (action_count / len(actions_taken)) * 100
                    print(f"  {i+1}. Action {action_value}: {action_count} times ({percentage:.1f}%)")
        
            # Multi-Episode Stats
            self.episode_rewards.append(episode_reward)
            min_episode_rewards = np.min(self.episode_rewards[-100:]) 
            max_episode_rewards = np.max(self.episode_rewards[-100:])
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            
            # Episode Stats
            ep_rewards = np.array(ep_reward_components)
            
            ## DeltaQ
            episode_delta_q = ep_rewards[:,0]
            ep_reward_deltaq = np.mean(episode_delta_q)
            ep_reward_min_deltaq = np.min(episode_delta_q)
            ep_reward_max_deltaq = np.max(episode_delta_q)
            
            ## Longwait
            episode_longwait = ep_rewards[:,1]
            ep_reward_longwait = np.mean(episode_longwait)
            ep_reward_min_longwait = np.min(episode_longwait)
            ep_reward_max_longwait = np.max(episode_longwait)
            
            episode_metrics = {}
                # Calculate episode-level metrics from step metrics (critical metrics only)
            if hasattr(self, 'episode_metrics') and self.episode_metrics:
                final_metrics = self.episode_metrics[-1]  # Get final state metrics
                
                # Only log critical traffic performance metrics
                episode_metrics = {
                    # Essential throughput metrics
                    'throughput_total': final_metrics.get('vehicles_exited_total', 0),
                    
                    # Essential waiting time metrics
                    'avg_waiting_time': final_metrics.get('avg_waiting_time', 0),
                    'vehicles_waiting_over_60s': final_metrics.get('vehicles_waiting_over_60s', 0),
                }
                
                # Reset episode metrics for next episode
                self.episode_metrics = []
                    
            if self.log_wandb:                         
                wandb.log({
                    'episode': episode+1,
                    'single_episode_reward': episode_reward,
                    'avg_reward (past 100 episodes)' : avg_reward,
                    'max_episode_reward' : max_episode_rewards,
                    'min_episode_reward' : min_episode_rewards,
                    'steps' : step_count,
                    'epsilon' : self.epsilon,
                    'episode_avg_deltaq' : ep_reward_deltaq,
                    'episode_max_deltaq' : ep_reward_max_deltaq,
                    'episode_min_deltaq' : ep_reward_min_deltaq,
                    'episode_avg_longwait' : ep_reward_longwait,
                    'episode_max_longwait' : ep_reward_max_longwait,
                    'episode_min_longwait' :ep_reward_min_longwait,
                    'actions_taken': wandb.Histogram(np.array(actions_taken)),
                    'actions_taken_sequence': actions_taken,
                    'action_changes' : action_changes,
                    **episode_metrics
                    })
                
            # Print episode summary with key metrics
            summary_text = f"\nEpisode {episode+1}:\nMoving Avg Reward (100 ep): {avg_reward:.2f}\nEpisode Reward: {episode_reward}\nEpsilon: {self.epsilon:.3f}\nStep Count: {step_count}"
            
            # Add throughput and waiting metrics to summary if available
            if 'throughput_total' in episode_metrics:
                summary_text += f"\nThroughput: {episode_metrics['throughput_total']} vehicles"
                summary_text += f"\nAvg Wait: {episode_metrics['avg_waiting_time']:.1f}s"
                summary_text += f" | Long Wait (>60s): {episode_metrics['vehicles_waiting_over_60s']}"
            
            print(summary_text + "\n")
            
            #Periodic evaluation
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=eval_episodes)
                print(f"[Eval] Episode {episode+1}: Average Evaluation Reward over {eval_episodes} episodes: {eval_reward:.2f}")
                if self.log_wandb:
                    wandb.log({'eval_avg_reward': eval_reward, 'eval_episode': episode+1})
                    
            # Reset environment after each episode
            self.end_episode()
            self.env.reset()
            # Save every 100 ep
            if (episode + 1) % 100 == 0:
                self.save(os.path.join(self.save_dir, f"{episode+1}.pth"))
        
            # Save trained model
        if episode + 1 != num_episodes:
            self.save(os.path.join(self.save_dir, "full.pth"))
        if self.log_wandb:
            wandb.finish()

    def run_without_training(self, render):
        """ Run a single episode without training, and tracks traffic metrics for comparison with SCATS model"""
        prev_action = None
        self.render = render   
        self.min_timer = self.config['min_action_timer']
        self.action_timer = 0
        self.hold_action = None
        total_qlength = np.zeros(12, dtype=float)
        counter = 0
        action_changes = 0
        ep_reward_components = []
        state = self.env._observe_NN()

        # Create GIF if render is true
        if self.render:
            date = datetime.now().strftime("%d%m%Y_%H%M%S")
            gif_filename = os.path.join(self.save_dir, f"DQN_{date}.gif")
            self.env.sumo.reset(gif=gif_filename)

        for _ in range(self.gym_config['max_simtime']): 
            action_idx = self.select_action(state, False)
            action = reasonable_actions[action_idx]
            next_state, reward, done, step_count, reward_components, step_metrics = self.env.step(action)
            ep_reward_components.append(reward_components)
            
            # qlength calculations for comparison with SCATS
            qlength = self.env.sumo.get_queue_length()
            total_qlength += qlength # sum of qlength after each step

            # total wait time of all cars registered by sensors
            step_waittime = self.env.sumo.get_occupied_time() # get wait time after each step for cars on sensors
            waittime = np.sum(step_waittime, axis=1) # sum wait time for each lane
            self.env.total_waittime += waittime
            
            # Throughput calculations for comparison with SCATS
            outgoing = self.env.sumo.get_left_intersection()
            self.env.throughput += outgoing

            # Update GIF frame if creating GIF
            if self.render:
                self.env.sumo._update_gif()
            prev_action = action

            # Sum results for comparison with SCATS model
            self.env.compare_reward += reward
            self.env.compare_deltaq += reward_components[0]
            self.env.compare_longwait += reward_components[1]
            state = next_state

            # counter to average queue length per time step
            counter += 1 
            if done:
                break

        self.env.average_qlength = total_qlength/counter # calculate average qlength over the course of 1 episode

        # Save GIF if one was created
        if self.render:
            self.env.sumo.save_gif()
            print(f"GIF saved: {gif_filename}")

    def _get_compare_rewards(self):
        """gets rewards for comparison"""
        return self.env.compare_reward, self.env.compare_deltaq, self.env.compare_longwait

    def _get_comparison_metrics(self):
        """gets metrics for comparison"""
        return self.env.average_qlength, self.env.total_waittime, self.env.throughput/self.gym_config['max_simtime']

    def _random(self):
        """randomised the sumo seed"""
        self.env.sumo.random_seed()

# Example usage
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
            "seed": 8
        },
        
        # Gym  Config
        'gym': {
            "max_simtime": 1800,
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
            'enabled': not args.no_wandb,
            'name': args.wandb_name
        }
    }

    sumo_config = unified_config['sumo']
    gym_config = unified_config['gym'] 
    agent_config = unified_config['agent']
    training_config = unified_config['training']

    # Create the Gym environment
    env = TrafficGym(sumo_config, config=gym_config)
    eval_env = TrafficGym(sumo_config, config=gym_config)

    # Environment parameters
    state_size = len(env._observe_NN())
    action_size = len(reasonable_actions)

    # Create agent with simplified constructor
    agent = DQNAgent(state_size, action_size, env, unified_config, eval_env,)
    # agent.load("./training/ky-test-34/1000.pth")

    num_episodes = training_config['num_episodes']

    agent.run(num_episodes)
    
# ==== HOW TO RUN =====
'''bash
pip install wandb
wandb login

To use log your test:
python environment/NN_wandb.py --wandb-name <testname>
 
If you dont want to use wandb:
python environment/NN_wandb.py --no-wandb
'''    


    