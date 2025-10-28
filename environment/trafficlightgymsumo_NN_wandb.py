import numpy as np
from sumo_interface import SumoInterface
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

class TrafficGym():

    def __init__(self, sumo_config, max_simtime, no_of_sensors, traffic_rate_upstream, traffic_rate_downstream,reward_weights):

        self.sensors = no_of_sensors

        self.sumo = SumoInterface(**sumo_config)     # Initialize SUMO interface

        self.upstream_status = traffic_rate_upstream        # "High", "Medium", "Low"

        self.downstream_status = traffic_rate_downstream        # "High", "Medium", "Low"
        
        self.ep_endtime = max_simtime # Max steps per episode
        self.apply_traffic_light = np.zeros(12, dtype=int)
        self.queue_length = np.zeros(12, dtype=int)
        self.occupied_time = np.zeros((12,self.sensors), dtype=float)
        self.done = False
        self.step_count = 0
        self.reward_weights = reward_weights

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
        self.prev_queue_length.fill(0)
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

        if self.downstream_status == "High":
            self.sumo.set_speed_slowdown([1.]*4)

        elif self.downstream_status == "Medium":
            self.sumo.set_speed_slowdown([0.5]*4)

        elif self.downstream_status == "Low":
            self.sumo.set_speed_slowdown([0.1]*4)

    def generate_rewards(self,reward_weights):
        
        w1 = reward_weights[0]
        w2 = reward_weights[1]
        # penalty_wait = 0
        penalty_longwait = 0
        
        delta_qlength = w1*int(self.prev_queue_length.sum() - self.queue_length.sum())

        # cars_waiting = self.occupied_time[self.occupied_time > 1]
        cars_waitinglong = np.sum(self.occupied_time>60)
        # if cars_waiting.size > 0:
        
        penalty_longwait = -w2*cars_waitinglong #*steptime

            # for cars_waiting in cars_waitinglong:
            #     penalty_longwait -= (w2*((cars_waiting/60)**(cars_waiting%60)))
            # else:
            #     penalty_wait = -(w2*np.mean(cars_waiting))
        
        delta_qlength = np.clip(delta_qlength / 10.0, -2, 2)

        total = delta_qlength + penalty_longwait

        return total, delta_qlength,penalty_longwait
  
    def step(self, action):
        
        self.step_count += 1
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
        self.new_state = self._observe_NN()
        
        # Load the action into rewards calculator
        # NOTE might want include upstream and downstream in reward somehow
        reward , delta_qlength, penalty_maxwait = self.generate_rewards(self.reward_weights)
        
        reward_components = [delta_qlength,penalty_maxwait]

        # # End episode if collision occurs
        # if self.collisions:
        #     #print("Vehicle collided.")
        #     self.done = True
        
        # End episode if max steps reached
        if self.simtime >= self.ep_endtime:
            #print(f"You have reached {self.max_steps}. Good job!")
            self.sumo.reset()
            self.done = True

        return self.new_state, reward, self.done, self.step_count, reward_components
    
    def _observe(self):
        """"
        Return observation state as a 1xfeature array for input to NN

        Traffic light state (12,1) = 12 values 
        Wait times (4,3,5) = 60 values 
        Current Q length (12,1) = 12 values
        Upstream, Downstream = 2 values
        Total = 86 values
        """
        self._get_state_from_sumo()

        if self.upstream_status == "High":
            upstream = 1

        elif self.upstream_status == "Medium":
            upstream = 0.5

        elif self.upstream_status == "Low":
            upstream = 0

        if self.downstream_status == "High":
            downstream = 1

        elif self.downstream_status == "Medium":
            downstream = 0.5

        elif self.downstream_status == "Low":
            downstream = 0

        return ([self.apply_traffic_light,self.occupied_time,self.queue_length,[upstream],[downstream]])

    def _observe_NN(self):
        """"
        Return observation state as a 1xfeature array for input to NN

        Traffic light state (12,1) = 12 values 
        Wait times (4,3,5) = 60 values 
        Current Q length (12,1) = 12 values
        Upstream - 1 values; High -1, Medium - 0.5, Low - 0
        Downstream - 1 values; High -1, Medium - 0.5, Low - 0
        Total = 86 values
        """
        next_state = self._observe()

        next_state[1] = np.clip(next_state[1]/120,0,1)
        
        next_state[2] = np.clip(next_state[2]/20,0,1)

        flatten_state = np.concatenate([np.array(state).flatten() for state in next_state])
    
        return flatten_state

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
    
         
    max_steps = 300     # CHANGE THIS (for max_simtime to end episode)
    queue_length = 5    # CHANGE THIS (for no. of induction loops on ground, max 5)
    traffic_rate_upstream = "Medium"
    traffic_rate_downstream = "Medium"
    reward_weights=[1,0.25]


    # Create the Gym environment
    env = TrafficGym(sumo_config, max_steps, queue_length, traffic_rate_upstream, traffic_rate_downstream,reward_weights)

    #lights = [1,1,1,0,0,0,1,1,1,0,0,0]      # NS Corridor is green
    #lights = [0,0,0,1,1,1,0,0,0,1,1,1]     # EW Corridor is green
    #action = encode_lights_to_action(lights) 
    
    action = 455
    rewards = []
    rewards_total = []

    for step in range(max_steps):
        if step % 20 == 0:
            action = np.random.randint(0, len(reasonable_actions))   # Random action from 0-4095 every 20 steps
        next_state, reward, done, step_count, reward_components = env.step(reasonable_actions[action])
        rewards.append(reward_components)
        rewards_total.append(reward)
        if step % 10 == 0:
#           env.sumo.visualize()
            np.set_printoptions(precision=2,suppress=True)
            
            # print(f"Step {step_count}: \nTime = {env.simtime} \nAction = {action} \nReward = {reward} \nReward Components = {reward_components} \nLight State = {env._observe()[0]}  \nOccupied Time = {env._observe()[1].reshape(4,3,5)} \nQueue Length per lane = {env._observe()[2].sum()} \nUpstream = {env._observe()[3]} \nDownstream = {env._observe()[4]} \nDone = {done}\n\n")
        if done:
            print(np.max(np.array(rewards)[:,0]),np.min(np.array(rewards)[:,0]))
            print(np.max(np.array(rewards)[:,1]),np.min(np.array(rewards)[:,1]))
            print(np.max(rewards_total),np.min(rewards_total))
            break

