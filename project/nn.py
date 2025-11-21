"""Defines the 2 main Neural Networks arrived at from project.

  Classes
NN - This is a standard DQN network
DuelingNN - This is a dueling DQN network
"""
import torch.nn as nn

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