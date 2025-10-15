import numpy as np
import gym
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
    
"""

class TrafficGym(gym.Env):

    def __init__(self, sumo_config, seed, max_steps, queue_length, traffic_rate_upstream, traffic_rate_downstream):

        self.queue_length = queue_length

        self.sumo = SumoInterface(**sumo_config)
        self.seed = seed
        self.randomspawn=np.random.RandomState(seed)

        self.apply_traffic_light = np.zeros(self.n_lights, dtype=int) #action to be taken

        self.upstream_status = traffic_rate_upstream #variable to spawn new vehicles in each lane
        self.downstream_status = traffic_rate_downstream #used for reward calculation
        
        self.max_steps = max_steps
        self.done = False
        self.step_count = 0

        self.lane_queue = np.zeros((4,3,self.queue_length), dtype=int)
        self.occupied_time = np.zeros((4,3,self.queue_length), dtype=float)
        self.collisions = 0
        self.time = 0

    def _get_state_from_sumo(self):

        '''
        Get observation from sumo:
        Output:
            - self.lane.queue: 4 x 3 x queue_length of 0s or 1s detecting presence of car on top of sensors at junction.
            - self.occupied_time: 4 x 3 x queue_length of floats indicating how long a car has been waiting at junction.
            - self.collisions: scalar of no. of collisions
            - self.time: time elapse since episode start
        '''

        # Retrieve occupancy from sensors from SUMO
        occupied = self.sumo.get_occupied()

        # Retrieve occupnacy time from sensors from SUMO
        occupied_time = self.sumo.get_occupied_time()
      
        self.lane_queue = occupied.reshape(4,3,self.queue_length)
        self.occupied_time = occupied_time.reshape(4,3,self.queue_length)

        self.cars_in_intersection = self.sumo.get_in_intersection()
        self.cars_left_intersection = self.sumo.get_left_intersection()

        self.collisions = self.sumo.get_collisions()
        self.time = self.sumo.get_time()

    def reset(self):
        # Reset local state variables
        self.apply_traffic_light.fill(0)
        self.lane_queue.fill(0)
        self.occupied_time.fill(0)
        self.cars_in_intersection.fill(0)
        self.cars_left_intersection.fill(0)

    def end_episode(self):
        self.done = True
        self.sumo.close()
        self.reset()
        print("Episode End.")

    def generate_rewards(self):
        queue = self.lane_queue.copy()
        traffic_state = self.apply_traffic_light.copy()
        reward_lane_movement = 0
        penalty_non_lane_movement = 0
        reward_entering_intersection = 0

        # Calculate rewards for lane movement and entering intersection, penalty for non lane movement
        for direction in range(4):
            for lane in range(3):
                for sensor in range(self.queue_length):
                    if sensor == 0 and queue[direction, lane, 0] == 1 and traffic_state[direction] == 1:
                        reward_entering_intersection += 3*self.upstream_status[direction]
                    elif sensor > 0 and queue[direction, lane, sensor] == 1 and queue[direction, lane, sensor - 1] == 0:
                        reward_lane_movement += self.upstream_status[direction]
                    if traffic_state[direction] == 0:
                        penalty_non_lane_movement -= (queue[direction, lane, :].sum())*self.upstream_status[direction]

        # Calculate penalty for vehicle remaining in intersection
        penalty_vehicle_in_intersection = -5 * np.sum(self.cars_in_intersection)

        # Calculate reward for clearing intersection, with bonus if downstream is clear
        reward_clearing_intersection = np.sum(self.cars_left_intersection * 10 * np.array(self.downstream_status))

        # Calculate penalty for collision
        penalty_collision = -20 * self.sumo.get_collisions()

        total = reward_lane_movement + penalty_non_lane_movement + reward_entering_intersection + penalty_vehicle_in_intersection + reward_clearing_intersection + penalty_collision

        return total

    def step(self, action):

        #Convert 0-4095 to 1x12
        self.apply_traffic_light = np.array([(action >> i) & 1 for i in range(12)])

        # Random spawn of cars from 4 directions
        car = self.randomspawn.choice(range(4), size=2, replace=False).astype(int)
        
        # Add car into SUMO
        self.sumo.add_car(*car)

        # Apply light to SUMO
        self.sumo.set_lights(self.apply_traffic_light)

        # Step SUMO
        self.sumo.step()

        # Get states from SUMO
        self._get_state_from_sumo()
        
        # Load the action into rewards calculator
        reward = self.generate_rewards()

        # New state after action
        new_state = np.array([self.lane_queue.flatten(),
                              self.apply_traffic_light.flatten(),
                              self.occupied_time.flatten(),
                              self.cars_in_intersection,
                              self.cars_left_intersection
                              ])

        # End episode if collision occurs
        if self.collisions:
            print("Vehicle collided.")
            self.end_episode()
        
        # End episode if max steps reached
        elif self.step_count == self.max_steps:
            print(f"You have reached {self.max_steps}. Good job!")
            self.end_episode()

        else:
            if self.step_count % 10 == 0:
                # env.sumo.visualize()
                print(f"Step {self.step_count}: \nAction={action}, \nReward={reward} \nTraffic before Intersection={new_state[0]}  \nLight State={new_state[1]} \nOccupied Time={new_state[2]} \nCars in Intersection={new_state[3]} \nCars left Intersection={new_state[4]}\n\n")

        self.step_count += 1
        
        return new_state,reward,env.done
    
'''
def _listNextValidActions(self, prev_action=0):
    available_actions = [
    0, #All Red
    455, #NS
    3640, #EW
    ]
'''

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

    # Gym config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--steps", type=int, default=300) #CHANGE THIS (for no. of steps you want to take)
    args = parser.parse_args()

    sumo_config = {
        "fname": "demo.sumocfg",
        #"gui": False,               # USE THIS (If you don't need to see the simulation)
        "gui": bool(args.gui),       # USE THIS (If you want to see simulation in SUMO)
        "cfg": {"directions": ["top0", "right0", "bottom0", "left0"]}
    }

    seed = 42           # CHANGE THIS (if you want a different spawn of cars)
    max_steps = 200     # CHANGE THIS (for max_steps to end episode)
    queue_length = 1    # CHANGE THIS (for no. of induction loops on ground)
    traffic_rate_upstream = [1, 1, 1, 1] 
    traffic_rate_downstream = [1, 1, 1, 1]

    # Create the Gym environment
    env = TrafficGym(sumo_config, seed, max_steps, queue_length, traffic_rate_upstream, traffic_rate_downstream)

    # Check environment
    print(f"Initial observation: \nTraffic before Intersection=\n{env.lane_queue} \nLight State={env.apply_traffic_light} \nOccupied Time=\n{env.occupied_time}")

    lights = [1,1,1,0,0,0,1,1,1,0,0,0] # CHANGE THIS (to change action sent to gym)
    action = encode_lights_to_action(lights) 

    for step in range(args.steps):
        env.step(action)
        if env.done:
            break