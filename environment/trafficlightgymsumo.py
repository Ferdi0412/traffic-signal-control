import numpy as np
import gym
import random
from sumo_interface import SumoInterface
from trafficlightrewards import TrafficLightRewards
import os
import sys
import time
import heapq
import argparse
from itertools import permutations

"""
1. Agent choose action (0-4095) 
2. Gym receive action, convert to 1x12 array, and send to SUMO
3. SUMO apply light config, sumo.step() advance by 1 timestep
4. In SUMO, vehicles move, collisions detected, new vehicles spawn in a random but repeatable manner
5. Gym reads SUMO's new state
    Observation:
    - Queue length for each direction
    - Intersection occupancy
    - Current light states
    
6. Gym offers reward    
    Reward:
    - Reward for lane movement
    - Penalty for non lane movement
    - Reward for entering intersection
    - Reward for clearing intersection
    - Penalty for vehicle remianing in intersection
    - Penalty for collision

7. Gym
    Valid Actions:
    - Determine next valid traffic light states

    Info:
    - No. of vehicles in system
    - Collision count
    - Timestep
"""

class TrafficGym(gym.Env):
    """
    Gym receive action, convert to 1x12 array

    Create observation space for SUMO's states
    - Queue length in each direction
    """

    def __init__(self, sumo_config, seed, queue_length,traffic_rate_upstream, traffic_rate_downstream):
        """
        Initialise variables to be used
        """
        self.n_lights = 12
        self.n_actions = 2**12
        self.queue_length = queue_length

        self.sumo = SumoInterface(**sumo_config)
        self.seed = seed
        self.randomspawn=np.random.RandomState(seed)

        self.prev_traffic_light = np.zeros(self.n_lights, dtype=int) #prev state of traffic light
        self.apply_traffic_light = np.zeros(self.n_lights, dtype=int) #action to be taken

        self.upstream_status = traffic_rate_upstream #variable to spawn new vehicles in each lane
        self.downstream_status = traffic_rate_downstream #used for reward calculation

        self.step_count = 0
    
    def _get_state_from_sumo(self):

        '''
        5. Gym reads SUMO's new state

        Input:
            12x1 array [NL,NF,NR,EL,EF,ER,SL,SF,SR,WL,WF,WR]
            NL:North Left
            NF:North Front
            NR:North Right
            E: East
            S: South
            W: West

        Output:
            4x3x1 array 
            [[NL,NF,NR],
             [EL,EF,ER],
             [SL,SF,SR],
             [WL,WF,WR]]

        Observation:
            - Queue length for each direction
            - Intersection occupancy (not done)
            - Current light states
            
        '''
        occupied = self.sumo.get_occupied()
        occupied_time = self.sumo.get_occupied_time()
        self.lane_queue = occupied.reshape(4,3,self.queue_length)
        self.occupied_time = occupied_time.reshape(4,3,self.queue_length)

        self.collisions = self.sumo.get_collisions
        self.step_time = self.sumo.get_step_time
        self.time = self.sumo.get_time

    def reset(self):
        """Reset to all states for a new episode"""
        # Reset local state variables
        self.prev_traffic_light.fill(0)
        self.apply_traffic_light.fill(0)
        self.step_count = 0
        
        # Read initial state from SUMO
        self._get_state_from_sumo()
        # Prepare initial observation
        initial_state = np.array([self.lane_queue.flatten(),
                              self.apply_traffic_light.flatten(),
                              self.occupied_time.flatten()])

        return initial_state

    def end_episode(self):
        """End episode if there is a collision"""
        self.reset()
        return

    def step(self, action):
        """
        Execute one action in SUMO, generate results

        1. Send 1x12 array to SUMO
        2. Advance SUMO by 1 timestep
        3. Read state from SUMO (observation)
        4. Calculate reward
        5. Return new state, reward

        Input:
            action (int) : Action number 0-4095 to get state from SUMO and calculate rewards

        Returns:
        """
        for i in range(self.n_lights):
            self.apply_traffic_light[i] = (action >> i) & 1

        #Apply light to sumo
        self.sumo.set_lights(self.apply_traffic_light)

        #Step SUMO
        self.sumo.step()

        self.sumo.visualize()

        #Get states from SUMO
        self._get_state_from_sumo()

        load = TrafficLightRewards(action, self.lane_queue,self.queue_length, self.upstream_status, self.downstream_status)
        load.step(action)
        reward = load.reward(action)
        new_state = np.array([self.lane_queue.flatten(),
                              self.apply_traffic_light.flatten(),
                              self.occupied_time.flatten()])
        
        return new_state,reward
    
def encode_lights_to_action(lights):
    #Convert 1x12 to 0-4095
    action = 0
    for i in range(12):
        if lights[i]:
            action |= (1 << i)
    return action

def decode_action_to_lights(action):
    #Convert 0-4095 to 1x12 
    return [(action >> i) & 1 for i in range(12)]

if __name__ == "__main__":

    sumo_config = {
        "fname": "demo.sumocfg",
        "gui": False,
        "cfg": {"directions": ["top0", "right0", "bottom0", "left0"]}
    }
    seed = 42
    queue_length = 1
    traffic_rate_upstream = [1, 1, 1, 1]
    traffic_rate_downstream = [1, 1, 1, 1]

    # Create the Gym environment
    env = TrafficGym(sumo_config, seed, queue_length, traffic_rate_upstream, traffic_rate_downstream)

    # Reset environment
    obs = env.reset()
    print("Initial observation:", obs)

    lights = [1,1,1,0,0,0,1,1,1,0,0,0]
    action = encode_lights_to_action(lights)

    for step in range(10):
        pts = env.randomspawn.choice(range(4), size=2, replace=False).astype(int)
        env.sumo.add_car(*pts)
        new_state, reward = env.step(action)
        print(f"Step {step}: \nAction={action}, \nReward={reward} \nTraffic before Intersection={new_state[0]}  \nLight State={new_state[1]} \nOccupied Time={new_state[2]}\n\n")
