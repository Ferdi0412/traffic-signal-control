import numpy as np
import gym
from .sumo_interface import SumoInterface
import argparse

"""
OVERALL LOGIC

1. Agent choose action (0-4095) 
2. Gym receive action, convert to 1x12 array, and send to SUMO
3. SUMO apply action, sumo.step() advance by 1 timestep
4. In SUMO, vehicles move, collisions detected, new vehicles spawn in a random but repeatable manner
5. Gym reads SUMO's new state
    Observation:
    - Queue length for each direction
    - Intersection occupancy
    - Current light states
    - Occupied time 
    
6. Gym offers reward    
    Reward:
    - Reward for lane movement
    - Penalty for non lane movement
    - Reward for entering intersection
    - Reward for clearing intersection
    - Penalty for vehicle remianing in intersection
    - Penalty for collision

7. Gym returns observation, reward, done

GENERAL ARRAY STRUCTURE

1x12 array for traffic lights: [NL,NF,NR,EL,EF,ER,SL,SF,SR,WL,WF,WR]
    
"""

class TrafficGym(gym.Env):

    def __init__(self, sumo_config, max_simtime, no_of_sensors, traffic_rate_upstream, traffic_rate_downstream):

        self.sensors = no_of_sensors

        self.sumo = SumoInterface(**sumo_config)     # Initialize SUMO interface

        self.apply_traffic_light = np.zeros(12, dtype=int) #action to be taken

        self.upstream_status = traffic_rate_upstream        # "High", "Medium", "Low"

        self.downstream_status = traffic_rate_downstream        # "High", "Medium", "Low"
        
        if traffic_rate_downstream == "High":
            self.sumo.set_speed_slowdown([1.]*4)

        elif traffic_rate_downstream == "Medium":
            self.sumo.set_speed_slowdown([0.5]*4)

        elif traffic_rate_downstream == "Low":
            self.sumo.set_speed_slowdown([0.1]*4)
        
        self.ep_endtime = max_simtime # Max steps per episode
        self.done = False 
        self.time = 0
        self.step_count = 0
        self.occupied_time = np.zeros((4,3,self.sensors), dtype=float) #initialise occupied time
        self.queue_length = np.zeros(12, dtype=int)

    def _get_state_from_sumo(self):

        # Retrieve occupnacy time from sensors from SUMO
        self.occupied_time = self.sumo.get_occupied_time()

        # Get queue length from SUMO
        self.queue_length = self.sumo.get_queue_length()
        
        # Reshape to 4 x 3 x queue_length
        # self.occupied_time = occupied_time.reshape(4,3,self.sensors)

        # Retrieve collisions and time elapsed since start of episode from SUMO
        self.collisions = self.sumo.get_collisions()
        self.simtime = self.sumo.get_time()

    def reset(self):
        # Reset local state variables
        self.done = False
        self.apply_traffic_light.fill(0)
        self.queue_length.fill(0)
        self.occupied_time.fill(0)
        self.collisions = 0
        self.time = 0 
        self.simtime = 0
        self.step_count = 0
        self.sumo.reset()
    
    def set_carspawn(self):
        if self.upstream_status == "High":
            self.sumo.set_car_prob([5 / 12] * 12)
        
        elif self.upstream_status == "Medium":
            self.sumo.set_car_prob([3 / 12] * 12)

        elif self.upstream_status == "Low":
            self.sumo.set_car_prob([1 / 12] * 12)

    def generate_rewards(self):
        total = 0 
        penalty_wait = 0
        
        delta_qlength = self.queue_length.sum() - self.prev_queue_length.sum()
    
        filtered_times = self.occupied_time[self.occupied_time > 1]
        
        # Only process if we have valid data
        if filtered_times.size > 0:  # .size is more NumPy-idiomatic than len()
            avg_wait = np.mean(filtered_times)
            max_wait = np.max(self.occupied_time)
            
            if avg_wait > 5:
                penalty_wait -= avg_wait * 0.05
            if max_wait > 60:
                penalty_wait -= max_wait - 60
        
        total += delta_qlength + penalty_wait
        return total,delta_qlength,penalty_wait
  
    def step(self, action):
        
        # Convert 0-4095 to 1x12
        # Index: 0-North Left,1-North Forward,2-North Right,3-EL,4-EF,5-ER,6-SL,7-SF,8-SR,9-WL,10-WF,11-WR
        
        self.apply_traffic_light = np.array([(action >> i) & 1 for i in range(12)])

        # Get queue length from SUMO before action
        self.prev_queue_length = self.sumo.get_queue_length()

        # Add car into SUMO
        self.set_carspawn()

        # Apply light to SUMO
        self.sumo.set_lights(self.apply_traffic_light)

        # Step SUMO
        self.sumo.step()

        # Get states from SUMO
        self._get_state_from_sumo()
        
        # Load the action into rewards calculator
        # NOTE might want include upstream and downstream in reward somehow
        reward , delta_qlength, penalty_wait= self.generate_rewards()
        
        # New state after action
        self.new_state = self._observe()

        # End episode if collision occurs
        if self.collisions:
            #print("Vehicle collided.")
            self.done = True
        
        # End episode if max steps reached
        elif self.simtime >= self.ep_endtime:
            #print(f"You have reached {self.max_steps}. Good job!")
            self.sumo.reset()
            self.done = True

        self.step_count += 1

        state_shape = ([self.new_state[0].shape, self.new_state[1].shape, self.new_state[2].shape])

        return self.new_state, reward, self.done, self.step_count, state_shape, delta_qlength, penalty_wait
    
    def _observe(self):
        """"
        Return observation state as a 1xfeature array for input to NN

        Traffic light state (12,1) = 12 values 
        Wait times (4,3,5) = 60 values 
        Current Q length (12,1) = 12 values

        Total = 84 values

        NOTE: Does not include
        Upstream state (4,1) = 4 values 
        Downstream state (4,1) = 4 values 
        """
        return ([self.apply_traffic_light,self.occupied_time,self.queue_length])

    def _observe_NN(self):
        """"
        Return observation state as a 1xfeature array for input to NN

        Traffic light state (12,1) = 12 values 
        Wait times (4,3,5) = 60 values 
        Current Q length (12,1) = 12 values
        Upstream - 1 values; High -2, Medium - 1, Low - 0
        Downstream - 1 values; High -2, Medium - 1, Low - 0
        Total = 86 values
        """

        _observation = np.array([])

        _observation = np.append(_observation, self.apply_traffic_light.flatten())

        _observation = np.append(_observation, self.occupied_time.flatten())

        _observation = np.append(_observation, self.queue_length.flatten())

        if self.upstream_status == "High":
            _observation = np.append(_observation, 2)

        elif self.upstream_status == "Medium":
            _observation = np.append(_observation, 1)

        elif self.upstream_status == "Low":
            _observation = np.append(_observation, 0)

        if self.downstream_status == "High":
            _observation = np.append(_observation, 2)

        elif self.downstream_status == "Medium":
            _observation = np.append(_observation, 1)

        elif self.downstream_status == "Low":
            _observation = np.append(_observation, 0)

        return _observation

'''
Implement valid actions
'''

def _listNextValidActions(self, prev_action=0):
    valid_actions = [
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

## === DEMO CODE ======================================================

if __name__ == "__main__":

    # Gym config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="map_2", help="SUMO file to use")
    parser.add_argument("-g", "--gui", action="store_true", help="Whether to show GUI")
    args = parser.parse_args()

    sumo_config = {
        "fname": args.file,             # CHANGE THIS (if you want to use a different map)
        #"gui": False,                  # USE THIS (If you don't need to see the simulation)
        "gui": args.gui,                # USE THIS (If you want to see simulation in SUMO),
        "seed": 42                      # CHANGE THIS (if you want a different spawn of cars
        }
    
         
    max_steps = 600     # CHANGE THIS (for max_simtime to end episode)
    queue_length = 5    # CHANGE THIS (for no. of induction loops on ground, max 5)
    traffic_rate_upstream = "Medium"
    traffic_rate_downstream = "Medium"

    # Create the Gym environment
    env = TrafficGym(sumo_config, max_steps, queue_length, traffic_rate_upstream, traffic_rate_downstream)

    #lights = [1,1,1,0,0,0,1,1,1,0,0,0]      # NS Corridor is green
    #lights = [0,0,0,1,1,1,0,0,0,1,1,1]     # EW Corridor is green
    #action = encode_lights_to_action(lights) 
    
    action = 455

    for step in range(max_steps):
        # if step % 20 == 0:
        #     action = np.random.randint(0, 4096)   # Random action from 0-4095 every 20 steps
        next_state, reward, done, step_count,next_state_shape,delta_qlength,penalty_wait = env.step(action)
        flatten = np.concatenate([state.flatten() for state in next_state])

        if step % 10 == 0:
#           env.sumo.visualize()
            print(f"Step {step_count}: \nTime = {env.simtime} \nAction = {action} \nReward = {reward} \nDelta Q = {delta_qlength} \nPenalty for waiting = {penalty_wait} \nLight State = {next_state[0]}  \nOccupied Time = {next_state[1].reshape(4,3,5)} \nQueue Length per lane = {next_state[2]} \nDone = {done} \nState Shape = {next_state_shape}\n\n")
        if done:
            break
