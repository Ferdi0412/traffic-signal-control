"""Some shared code between the various agent files/NN files.

  Contains
ReplayBuffer
reasonable_cations
"""

import numpy as np
import random
from collections import deque



class ReplayBuffer:
    """Experience replay buffer for storing transitions
    
      Entry Format
    0: state
    1: action
    2: reward
    3: next_state
    4: done
    """
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



reasonable_actions = [
    3,     # North Left+Forward
    4,     # North Right Only
    7,     # North All
    24,    # East Left+Forward
    32,    # East Right Only
    56,    # East All
    192,   # South Left+Forward
    195,   # North Left+Forward + South Left+Forward
    196,   # North Right + South Left+Forward
    199,   # North All + South Left+Forward
    256,   # South Right Only
    259,   # North Left+Forward + South Right
    260,   # North Right + South Right
    263,   # North All + South Right
    448,   # South All
    451,   # North Left+Forward + South All
    452,   # North Right + South All
    455,   # North All + South All
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
    3640   # East All + West All
]