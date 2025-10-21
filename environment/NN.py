import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from trafficlightgymsumo_NN import TrafficGym
from sumo_interface import SumoInterface
import argparse
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class NN(nn.Module):
    """Deep Q-Network for traffic signal control"""
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()

        # Flatten input layer to 1D
        self.flatten = nn.Flatten()

        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(state_size)

        # Define network layers
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_size)
        )
    
    def forward(self, x):
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
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        default_config = {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'batch_size': 64,
            'target_update_freq': 10,
            'hidden_sizes': [128, 128]
        }
        self.config = {**default_config, **(config or {})}
        
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
        
        # Replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'])
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # # Decay epsilon
        # if self.epsilon > self.config['epsilon_min']:
        #     self.epsilon *= self.config['epsilon_decay']
        
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
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {filepath}")



# Example usage
if __name__ == "__main__":
    # Environment parameters
    state_size = 140  # placeholder input state
    action_size = 4096 # all possible actions

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="map_2", help="SUMO file to use")
    parser.add_argument("-g", "--gui", action="store_true", help="Whether to show GUI")
    parser.add_argument("-r", "--reset", action="store_true", help="Reset for 2 'playthroughs'")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    sumo_config = {
        "fname": args.file,             # CHANGE THIS (if you want to use a different map)
        #"fname": "demo.sumocfg",
        #"gui": False,                  # USE THIS (If you don't need to see the simulation)
        "gui": args.gui,                # USE THIS (If you want to see simulation in SUMO),
        }
    
    seed = 42           # CHANGE THIS (if you want a different spawn of cars)
    max_steps = 200     # CHANGE THIS (for max_steps to end episode)
    queue_length = 5    # CHANGE THIS (for no. of induction loops on ground, max 5)
    traffic_rate_upstream = [1, 1, 1, 1] 
    traffic_rate_downstream = [1, 1, 1, 1]

    # Create the Gym environment
    env = TrafficGym(sumo_config, seed, max_steps, queue_length, traffic_rate_upstream, traffic_rate_downstream)
    
    action = 0

    for step in range(args.steps):
        if step % 20 == 0:
            action = np.random.randint(0, 4096)   # Random action from 0-4095 every 20 steps
        env.step(action)
        if env.done:
            break

    # Create agent
    config = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'hidden_sizes': [256, 128]
    }
    agent = DQNAgent(state_size, action_size, config)
    
    # Training loop example
    num_episodes = 1000
    max_steps = 200
    
    for episode in range(num_episodes):
        action = np.random.randint(0, 4096)   # Initialise a random action to begin each episode
        episode_reward = 0
        state = env._observe()

        for step in range(max_steps):  
            env.step(action)
            next_state = env._observe()
            action = agent.select_action(next_state)
            reward = env.generate_rewards()
            done = step == max_steps - 1
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.episode_rewards.append(episode_reward)
        
        avg_reward = np.mean(agent.episode_rewards[-100:])
        print(f"\nEpisode {episode}:\nMoving Avg Reward (100 ep): {avg_reward:.2f}\nEpsilon: {agent.epsilon:.3f}\nLoss: {loss:.3f}\n")
        
        # Reset environment after each episode
        agent.end_episode()
        env.reset()
    
    # Save trained model
    #agent.save("traffic_dqn_model.pth")