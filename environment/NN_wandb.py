import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from trafficlightgymsumo_NN_wandb import TrafficGym
import argparse
import wandb
import os

reasonable_actions = [
0,  # All Red (Transition)
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
    """Deep Q-Network for traffic signal control"""
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()

        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(state_size)

        # Define network layers
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        #self.fc1 = nn.Linear(state_size, 128)        self.bn1 = nn.BatchNorm1d(128)  # Normalize layer outputs        self.fc2 = nn.Linear(128, 128)        self.bn2 = nn.BatchNorm1d(128)        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        if x.shape[0] > 1:
            x = self.input_bn(x)
        return self.network(x)

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
    def __init__(self, state_size, action_size, env, eval_env, config=None, log_wandb = True, name=None):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.eval_env = eval_env
        
        self.log_wandb = log_wandb
        self.wandb_testname = name
        if self.wandb_testname:
            self.save_dir = os.path.join("./training", self.wandb_testname)
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = "./"
        
        self.config = {**(config or {})}
        
        self.epsilon = self.config['epsilon']
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.target_update_freq = self.config['target_update_freq']
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = NN(state_size, action_size).to(self.device)
        self.target_net = NN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # Replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'])
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []

        if log_wandb:
            wandb.init(
                entity = "kaiyi-lam-ml",
                project="traffic-signal-control",
                name= name,
                config={
                    **config,
                }
            )
            wandb.watch(self.policy_net, log='all', log_freq=100)

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""        
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the DQN using experience replay"""
        if len(self.memory) < self.batch_size:
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
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Find gradient norm
        grad_norm = 0.
        for policy in self.policy_net.parameters():
            if policy.grad is not None:
                grad_norm += policy.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        #Clip gradient
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        if self.training_step % 100 == 0:  # Update LR every 100 training steps
            self.scheduler.step()
            
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        td_error = torch.abs(target_q - current_q).mean().item()

        if self.log_wandb:
            wandb.log({
                'loss': loss.item(), #decreasing
                'mean_q_value': current_q.mean().item(), #should increase
                'td_error': td_error, #decreasing but not too low
                'grad_norm': grad_norm, #stable in 0.1-10 range
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
            for _ in range(600):
                action_idx = self.select_action(state, training=False)
                action = reasonable_actions[action_idx]
                next_state, reward, done, step_count, reward_components = eval_env.step(action)
                eval_reward_components.append(reward_components)
                episode_reward += reward
                state = next_state
                if done:
                    break
            eval_rewards.append(episode_reward)
            self.eval_env.reset()

        return np.mean(eval_rewards)

    def run(self, num_episodes, eval_interval=50, eval_episodes=5):
        for episode in range(num_episodes):
            action = np.random.randint(0, self.action_size)   # Initialise a random action to begin each episode
            episode_reward = 0.
            ep_reward_components = []
            actions_taken = []
            state = self.env._observe_NN()
            for _ in range(self.env.ep_endtime):
                action_idx = self.select_action(state)
                action = reasonable_actions[action_idx]
                next_state, reward, done, step_count, reward_components = self.env.step(action)
                ep_reward_components.append(reward_components)
                actions_taken.append(action)
                self.store_transition(state, action_idx, reward, next_state, done)
                loss = self.train()
                episode_reward += reward
                state = next_state
                if done:
                    break
            self.episode_rewards.append(episode_reward)
            ep_rewards = np.array(ep_reward_components) 
            ep_reward_deltaq = np.mean(ep_rewards[:,0])
            ep_reward_waitinglong = np.mean(ep_rewards[:,1])
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            
            if self.log_wandb:
                wandb.log({
                    'episode': episode+1,
                    'episode_reward': episode_reward,
                    'avg_reward' : avg_reward,
                    'steps' : step_count,
                    'epsilon' : self.epsilon,
                    'ep_avg_deltaq' : ep_reward_deltaq,
                    'ep_avg_longwait' : ep_reward_waitinglong,
                    'actions_taken': wandb.Histogram(np.array(actions_taken)),
                    'actions_taken_sequence': actions_taken,
                    })
            print(f"\nEpisode {episode+1}:\nMoving Avg Reward (100 ep): {avg_reward:.2f}\nEpsilon: {self.epsilon:.3f}\nStep Count: {step_count}\n")
            # Periodic evaluation
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
        self.save(os.path.join(self.save_dir, "full.pth"))
        if self.log_wandb:
            wandb.finish()

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="map_2", help="SUMO file to use")
    parser.add_argument("-g", "--gui", action="store_true", help="Whether to show GUI")
    parser.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    sumo_config = {
        "fname": args.file,             # CHANGE THIS (if you want to use a different map)
        #"gui": False,                  # USE THIS (If you don't need to see the simulation)
        "gui": args.gui,                # USE THIS (If you want to see simulation in SUMO),
        "seed": 88                      # CHANGE THIS (if you want a different spawn of cars
        }
         
    max_steps = 3600    # CHANGE THIS (for max_simtime to end episode)
    queue_length = 5    # CHANGE THIS (for no. of induction loops on ground, max 5)
    traffic_rate_upstream = "Medium"
    traffic_rate_downstream = "Medium"
    reward_weights = [1,0]

    # Create the Gym environment
    env = TrafficGym(sumo_config, max_steps, queue_length, traffic_rate_upstream, traffic_rate_downstream,reward_weights)
    eval_env = TrafficGym(sumo_config, max_steps, queue_length, traffic_rate_upstream, traffic_rate_downstream,reward_weights)
    
    # Environment parameters
    state_size = len(env._observe_NN())
    action_size = len(reasonable_actions)

    # Config
    config = {
        'learning_rate': 0.00001,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update_freq': 500
    }

    log_wandb = not args.no_wandb
    
    agent = DQNAgent(state_size, action_size, env, eval_env, config, log_wandb, args.wandb_name)
    
    # Training loop example
    num_episodes = 1000

    agent.run(num_episodes)
    


    