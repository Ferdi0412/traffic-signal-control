import numpy as np
import gym
import random

class TrafficLightRewards(object):
    """
    Direct control of 12 traffic lights.
    Agent can set any combination of lights using action numbers 0-4095.
    """

    def __init__(self, action, occupancy, queue_length, traffic_rate_upstream, traffic_rate_downstream):
        """
        Initialise variables to be used
        """
        self.n_lights = 12
        self.n_actions = 2**12
        self.queue_length = queue_length
        self.reward_for_lane_movement = np.zeros(4096, dtype=float)
        self.penalty_for_non_lane_movement = np.zeros(4096, dtype=float)
        self.reward_for_entering_intersection = np.zeros(4096, dtype=float)
        self.reward_for_clearing_intersection = np.zeros(4096, dtype=float)
        self.penalty_for_vehicle_remaining_instersection = np.zeros(4096, dtype=float)
        self.penalty_for_collision = np.zeros(4096, dtype=float)
        self.upstream_status = traffic_rate_upstream #variable to spawn new vehicles in each lane
        self.downstream_status = traffic_rate_downstream #used for reward calculation

        self.traffic_state = self.traffic_state = np.array([(action >> i) & 1 for i in range(self.n_lights)], dtype=int)
        
        # N: 0; S: 1; E: 1; W: 1 for the lane before/after junction array
        self.lane_before_junction = occupancy
        self.lane_after_junction = np.zeros((4,3,self.queue_length), dtype=int) #vehicle occupancy in lane after junction
        self.intersection = np.zeros((6,6), dtype=int) #vehicle occupancy in junction
        self.intersection_dir = np.full((6,6), " ", dtype=str) #direction of vehicle in junction
    
    def reset(self):
        """Reset to all states"""
        self.traffic_state.fill(0)
        self.lane_before_junction.fill(0)
        self.lane_after_junction.fill(0)
        self.intersection.fill(0)
        self.intersection_dir[:,:] = " "

    def end_episode(self):
        """End episode if there is a collision"""
        self.reset()
        return

    def step(self, action):
        """
        Apply an action (0-4095) to set all 12 lights.
        
        Move vehicles based on selected action
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions-1}")
        
        # Convert action input from int to binary form in state as an array
        for i in range(self.n_lights):
            self.traffic_state[i] = (action >> i) & 1
        
        # agent_state[0] - North Left Traffic Light
        # agent_state[1] - North Straight Traffic Light
        # agent_state[2] - North Right Traffic Light
        # agent_state[3] - South Left Traffic Light
        # agent_state[4] - South Straight Traffic Light
        # agent_state[5] - South Right Traffic Light
        # agent_state[6] - East Left Traffic Light
        # agent_state[7] - East Straight Traffic Light
        # agent_state[8] - East Right Traffic Light
        # agent_state[9] - West Left Traffic Light
        # agent_state[10] - West Straight Traffic Light
        # agent_state[11] - West Right Traffic Light
        
        self.move_veh_after_junction()
        self.move_veh_in_junction(action)
        self.move_veh_before_junction_east(action)
        self.move_veh_before_junction_north(action)
        self.move_veh_before_junction_south(action)
        self.move_veh_before_junction_west(action)
        self.generate_new_veh_upstream()

        return (self.traffic_state.copy(), 
                self.lane_before_junction.copy(), 
                self.lane_after_junction.copy(),
                self.intersection.copy(), 
                self.intersection_dir.copy())

    def move_veh_after_junction(self):
        """
        Move vehicles after junction downstream
        """
        for i in range(4):
            for j in range(3):
                for k in range(self.queue_length):
                    if k == 0 and self.lane_after_junction[i, j, 0] == 1:
                        self.lane_after_junction[i, j, 0] = 0 #vehicle reaches end of road and exit downstream
                        self.downstream_status[i] += 1/self.queue_length #traffic exiting lane contribute to downstream traffic
                    elif k > 0 and self.lane_after_junction[i, j, k] == 1 and self.lane_after_junction[i, j, k - 1] == 0:
                        self.lane_after_junction[i, j, k - 1] = 1 #vehicle moves one step forward
                        self.lane_after_junction[i, j, k] = 0
                if random.uniform(0,1) <= self.downstream_status[i]:
                            self.downstream_status[i] -= 1/self.queue_length #probability of traffic clearing downstream based on defined rate

    def move_veh_in_junction(self, action):
        """
        Move vehicles in junction and update the direction array
        """
        temp_intersection_dir = np.full((6,6), " ", dtype=str) # store directions of vehicles temporary, to prevent directions of other vehicle from being overwritten
        #move vehicles based on a 6x6 logic
        for i in range(6):
            for j in range(6):
                if self.intersection_dir[i,j] == "N":
                    if j < 3 and i !=0:
                        self.intersection[i, j] -= 1
                        self.intersection[i - 1, j] += 1
                        temp_intersection_dir[i - 1, j] = "N"
                        self.intersection_dir[i, j] = " "
                    elif i == 2 and j == 3:
                        self.intersection[2, 3] -= 1
                        self.intersection[1, 2] += 1
                        temp_intersection_dir[1, 2] = "N"
                        self.intersection_dir[2, 3] = " "
                    elif i == 3 and j == 4:
                        self.intersection[3, 4] -= 1
                        self.intersection[2, 3] += 1
                        temp_intersection_dir[2, 3] = "N"
                        self.intersection_dir[3, 4] = " "
                    elif i == 3 and j == 5:
                        self.intersection[3, 5] -= 1
                        self.intersection[3, 4] += 1
                        temp_intersection_dir[3, 4] = "N"
                        self.intersection_dir[3, 5] = " "
                    elif i == 0:
                        self.intersection[0,j] -= 1
                        temp_intersection_dir[0,j] = " "
                        self.lane_after_junction[0, j, self.queue_length - 1] = 1
                        self.reward_for_clearing_intersection[action] += 10*self.downstream_status[0]
                if self.intersection_dir[5 - i, 5 - j] == "S":
                    if j < 3 and i != 0:
                        self.intersection[5 - i, 5 - j] -= 1
                        self.intersection[5 - i + 1, 5 - j] += 1
                        temp_intersection_dir[5 - i + 1, 5 - j] = "S"
                        self.intersection_dir[5 - i, 5 - j] = " "
                    elif i == 2 and j == 3:
                        self.intersection[3,2] -= 1
                        self.intersection[4,3] += 1
                        temp_intersection_dir[4,3] = "S"
                        self.intersection_dir[3,2] = " "
                    elif i == 3 and j == 4:
                        self.intersection[2, 1] -= 1
                        self.intersection[3, 2] += 1
                        temp_intersection_dir[3, 2] = "S"
                        self.intersection_dir[2, 1] = " "
                    elif i == 3 and j == 5:
                        self.intersection[2, 0] -= 1
                        self.intersection[2, 1] += 1
                        temp_intersection_dir[2, 1] = "S"
                        self.intersection_dir[2, 0] = " "
                    elif i == 0:
                        self.intersection[5, 5 - j] -= 1
                        temp_intersection_dir[5, 5 - j] = " "
                        self.lane_after_junction[1, j, self.queue_length - 1] = 1
                        self.reward_for_clearing_intersection[action] += 10*self.downstream_status[1]
                if self.intersection_dir[i, 5 - j] == "E":
                    if i < 3 and j != 0:
                        self.intersection[i, 5 - j] -= 1
                        self.intersection[i, 5 - j + 1] += 1
                        temp_intersection_dir[i, 5 - j + 1] = "E"
                        self.intersection_dir[i, 5 - j] = " "
                    elif i == 3 and j == 2:
                        self.intersection[3,3] -= 1
                        self.intersection[2,4] += 1
                        temp_intersection_dir[2,4] = "E"
                        self.intersection_dir[3,3] = " "
                    elif i == 4 and j == 3:
                        self.intersection[4,2] -= 1
                        self.intersection[3,3] += 1
                        temp_intersection_dir[3,3] = "E"
                        self.intersection_dir[4,2] = " "
                    elif i == 5 and j == 3:
                        self.intersection[5,2] -= 1
                        self.intersection[4,2] += 1
                        temp_intersection_dir[4,2] = "E"
                        self.intersection_dir[5,2] = " "
                    elif j == 0 and i < 3:
                        self.intersection[i,5] -= 1
                        temp_intersection_dir[i,5] = " "
                        self.lane_after_junction[2, i, self.queue_length - 1] = 1
                        self.reward_for_clearing_intersection[action] += 10*self.downstream_status[2]
                if self.intersection_dir[5 - i, j] == "W":
                    if i < 3 and j != 0:
                        self.intersection[5 - i, j] -= 1
                        self.intersection[5 - i, j - 1] += 1
                        temp_intersection_dir[5 - i, j - 1] = "W"
                        self.intersection_dir[5 - i, j] = " "
                    elif i == 3 and j == 2:
                        self.intersection[2,2] -= 1
                        self.intersection[3,1] += 1
                        temp_intersection_dir[3,1] = "W"
                        self.intersection_dir[2,2] = " "
                    elif i == 4 and j == 3:
                        self.intersection[1, 3] -= 1
                        self.intersection[2, 2] += 1
                        temp_intersection_dir[2, 2] = "W"
                        self.intersection_dir[1, 3] = " "
                    elif i == 5 and j == 3:
                        self.intersection[0, 3] -= 1
                        self.intersection[1, 3] += 1
                        temp_intersection_dir[1, 3] = "W"
                        self.intersection_dir[0, 3] = " "
                    elif j == 0 and i < 3:
                        self.intersection[5 - i, 0] -= 1
                        temp_intersection_dir[5 - i, 0] = " "
                        self.lane_after_junction[3, i, self.queue_length - 1] = 1
                        self.reward_for_clearing_intersection[action] += 10*self.downstream_status[3]
        self.intersection_dir = temp_intersection_dir[:,:]
        self.penalty_for_vehicle_remaining_instersection -= 5*self.intersection.sum()
        #check for collision, if individual cell value is 2 or more, indicates a collision and ends episode
        for i in range(6):
            for j in range(6):
                if self.intersection[i,j] > 1:
                    self.penalty_for_collision[action] -= 20
                    self.end_episode()

    def move_veh_before_junction_north(self, action):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions-1}")
            agent_state = encode_state_to_action(action)
        traffic_number = 0 # counter for traffic light in for loop based on traffic light definition
        for i in range(3):
            for j in range(self.queue_length):
                if j == 0 and self.lane_before_junction[0, i, 0] == 1 and self.traffic_state[traffic_number] == 1:
                    self.lane_before_junction[0, i, 0] = 0 # vehicle enters intersection
                    self.intersection[5 , i] += 1
                    self.reward_for_entering_intersection[action] += 3*self.upstream_status[0]
                    if i == 0:
                        self.intersection_dir[5, 0] = random.choices(["W", "N"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn left
                    elif i == 1:
                        self.intersection_dir[5, 1] = "N"
                    elif i == 2:
                        self.intersection_dir[5, 2] = random.choices(["E", "N"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn right
                elif j > 0 and self.lane_before_junction[0, i, j] == 1 and self.lane_before_junction[0, i, j - 1] == 0:
                    self.lane_before_junction[0, i, j - 1] = 1 # vehicle moves forward in lane
                    self.lane_before_junction[0, i, j] = 0
                    self.reward_for_lane_movement[action] += self.upstream_status[0]
                if self.traffic_state[traffic_number] == 0:
                    self.penalty_for_non_lane_movement[action] -= (self.lane_before_junction[0, i, :].sum())*self.upstream_status[0]
            traffic_number += 1

    def move_veh_before_junction_south(self, action):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions-1}")
            agent_state = encode_state_to_action(action)
        traffic_number = 3 # counter for traffic light in for loop based on traffic light definition
        for i in range(3):
            for j in range(self.queue_length):
                if j == 0 and self.lane_before_junction[1, i, 0] == 1 and self.traffic_state[traffic_number] == 1:
                    self.lane_before_junction[1, i, 0] = 0 # vehicle enters intersection
                    self.intersection[0 , i + 3] += 1
                    self.reward_for_entering_intersection[action] += 3*self.upstream_status[1]
                    if i == 0:
                        self.intersection_dir[0, 3] = random.choices(["W", "S"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn left
                    elif i == 1:
                        self.intersection_dir[0, 4] = "S"
                    elif i == 2:
                        self.intersection_dir[0, 5] = random.choices(["E", "S"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn right
                elif j > 0 and self.lane_before_junction[1, i, j] == 1 and self.lane_before_junction[1, i, j - 1] == 0:
                    self.lane_before_junction[1, i, j - 1] = 1 # vehicle moves forward in lane
                    self.lane_before_junction[1, i, j] = 0
                    self.reward_for_lane_movement[action] += self.upstream_status[1]
                if self.traffic_state[traffic_number] == 0:
                    self.penalty_for_non_lane_movement[action] -= (self.lane_before_junction[1, i, :].sum())*self.upstream_status[1]
            traffic_number += 1

    def move_veh_before_junction_east(self, action):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions-1}")
            agent_state = encode_state_to_action(action)
        traffic_number = 6 # counter for traffic light in for loop based on traffic light definition
        for i in range(3):
            for j in range(self.queue_length):
                if j == 0 and self.lane_before_junction[2, i, 0] == 1 and self.traffic_state[traffic_number] == 1:
                    self.lane_before_junction[2, i, 0] = 0 # vehicle enters intersection
                    self.intersection[i , 0] += 1
                    self.reward_for_entering_intersection[action] += 3*self.upstream_status[2]
                    if i == 0:
                        self.intersection_dir[0, 0] = random.choices(["N", "E"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn left
                    elif i == 1:
                        self.intersection_dir[1, 0] = "E"
                    elif i == 2:
                        self.intersection_dir[2, 0] = random.choices(["S", "E"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn right
                elif j > 0 and self.lane_before_junction[2, i, j] == 1 and self.lane_before_junction[2, i, j - 1] == 0:
                    self.lane_before_junction[2, i, j - 1] = 1 # vehicle moves forward in lane
                    self.lane_before_junction[2, i, j] = 0
                    self.reward_for_lane_movement[action] += self.upstream_status[2]
                if self.traffic_state[traffic_number] == 0:
                    self.penalty_for_non_lane_movement[action] -= (self.lane_before_junction[2, i, :].sum())*self.upstream_status[2]
            traffic_number += 1

    def move_veh_before_junction_west(self, action):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions-1}")
            agent_state = encode_state_to_action(action)
        traffic_number = 11 # counter for traffic light in for loop based on traffic light definition
        for i in range(3):
            for j in range(self.queue_length):
                if j == 0 and self.lane_before_junction[3, i, 0] == 1 and self.traffic_state[traffic_number] == 1:
                    self.lane_before_junction[3, i, 0] = 0 # vehicle enters intersection
                    self.intersection[i + 3, 5] += 1
                    self.reward_for_entering_intersection[action] += 3*self.upstream_status[3]
                    if i == 0:
                        self.intersection_dir[3, 5] = random.choices(["N", "W"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn left
                    elif i == 1:
                        self.intersection_dir[4, 5] = "W"
                    elif i == 2:
                        self.intersection_dir[5, 5] = random.choices(["S", "W"], weights=[0.5, 0.5])[0] # 50% chance for vehicle to go straight or turn right
                elif j > 0 and self.lane_before_junction[3, i, j] == 1 and self.lane_before_junction[3, i, j - 1] == 0:
                    self.lane_before_junction[3, i, j - 1] = 1 # vehicle moves forward in lane
                    self.lane_before_junction[3, i, j] = 0
                    self.reward_for_lane_movement[action] += self.upstream_status[3]
                if self.traffic_state[traffic_number] == 0:
                    self.penalty_for_non_lane_movement[action] -= (self.lane_before_junction[3, i, :].sum())*self.upstream_status[3]
            traffic_number -= 1
    
    def generate_new_veh_upstream(self):
        """
        Spawn new vehicles into lane based on upstream status (probability)
        """
        for i in range(4):
            for j in range(3):
                if random.uniform(0,1) <= self.upstream_status[i] and self.lane_before_junction[i, j, self.queue_length - 1] == 0:
                    self.lane_before_junction[i, j, self.queue_length - 1] = 1

    def reward(self, action):
        """
        Compute reward sum and returns reward for a specific action state
        """
        reward_sum = np.zeros(4096, dtype=float)
        reward_sum = self.reward_for_lane_movement + self.penalty_for_non_lane_movement + self.reward_for_entering_intersection + self.reward_for_clearing_intersection + self.penalty_for_vehicle_remaining_instersection + self.penalty_for_collision
        return reward_sum[action]


    def decode_action(self, action):
        """
        Convert action number to light pattern without changing state.
        
        Args:
            action (int): Action number (0-4095)
            
        Returns:
            list: Light pattern [0/1, 0/1, ..., 0/1]
        """
        return [(action >> i) & 1 for i in range(self.n_lights)]
    
    def encode_state_to_action(self, state=None):
        """
        Convert a state pattern to its corresponding action number.
        
        Args:
            state (list/array): Light pattern. If None, uses current state.
            
        Returns:
            int: Action number that would produce this state
            
        Example:
            state = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
            returns: 3315 (binary: 110011001100)
        """
        if state is None:
            state = self.traffic_state
        
        action = 0
        for i in range(self.n_lights):
            if state[i]:
                action |= (1 << i)  # Set bit i
        return action
    
    def get_state(self):
        """Get current state"""
        return self.traffic_state.copy()
    
    def get_action_space_size(self):
        """Return total number of possible actions"""
        return self.n_actions


class demo_env:
    """
    Generate a random episode
    """
    def __init__(self, env, queue_length):
        
        self.traffic_light = env
        self.queue_length = queue_length

        # Initialize road states
        self.east_before_junc = np.zeros((3, queue_length), dtype=int)
        self.east_after_junc = np.zeros((3, queue_length), dtype=int)
        self.west_before_junc = np.zeros((3, queue_length), dtype=int)
        self.west_after_junc = np.zeros((3, queue_length), dtype=int)
        self.north_before_junc = np.zeros((queue_length, 3), dtype=int)
        self.north_after_junc = np.zeros((queue_length, 3), dtype=int)
        self.south_before_junc = np.zeros((queue_length, 3), dtype=int)
        self.south_after_junc = np.zeros((queue_length, 3), dtype=int)

        # Initialize junction states
        self.junction = np.zeros((6, 6), dtype=int)
        self.junction_dir = np.full((6, 6), " ", dtype=str)

        # Helper function to create border-marked quadrant
        def make_quadrant(rows, cols, border_i=None, border_j=None):
            arr = np.full((rows, cols), " ", dtype=str)
            if border_i is not None:
                arr[border_i, :] = "*"
            if border_j is not None:
                arr[:, border_j] = "*"
            return arr

        q = queue_length + 1
        self.top_left  = make_quadrant(q, q, border_i=queue_length, border_j=queue_length)
        self.top_right = make_quadrant(q, q, border_i=queue_length, border_j=0)
        self.btm_left  = make_quadrant(q, q, border_i=0, border_j=queue_length)
        self.btm_right = make_quadrant(q, q, border_i=0, border_j=0)

        self.agent_state = np.zeros(12, dtype=int)

    def reset(self):
        self.current_phase = self.traffic_light.reset()
        self.agent_state = np.zeros(12, dtype=int)
        # Reset counters
        self.current_step = self.vehicles_cleared = self.total_steps = 0

    def step(self, action):
        agent_state, lane_before_junc, lane_after_junc, intersection, intersection_dir = self.traffic_light.step(action)
        
        #format vehicle occupancy in lane to format for printing in demo_env class
        self.east_before_junc = lane_before_junc[2, :, :]
        self.east_before_junc = self.east_before_junc[:,::-1]
        self.east_after_junc = lane_after_junc[2, :, :]
        self.east_after_junc = self.east_after_junc[:,::-1]
        self.west_before_junc = lane_before_junc[3, :, :]
        self.west_after_junc = self.west_after_junc[::-1,:]
        self.west_after_junc = lane_after_junc[3, :, :]
        self.west_after_junc = self.west_after_junc[::-1,:]
        self.north_before_junc = lane_before_junc[0, :, :]
        self.north_before_junc = self.north_before_junc.T
        self.north_after_junc = lane_after_junc[0, :, :]
        self.north_after_junc = self.north_after_junc.T
        self.south_before_junc = lane_before_junc[1, :, :]
        self.south_before_junc = self.south_before_junc.T
        self.south_before_junc = self.south_before_junc[::-1,:]
        self.south_before_junc = self.south_before_junc[:,::-1]
        self.south_after_junc = lane_after_junc[1, :, :]
        self.south_after_junc = self.south_after_junc.T
        self.south_after_junc = self.south_after_junc[::-1,:]
        self.south_after_junc = self.south_after_junc[:,::-1]
        self.junction = intersection[:,:]
        self.junction_dir = intersection_dir[:,:]
        self.agent_state = agent_state

    def encode_state_to_action(self, state=None):
            """
            Convert a state pattern to its corresponding action number.
            
            Args:
                state (list/array): Light pattern. If None, uses current state.
                
            Returns:
                int: Action number that would produce this state
                
            Example:
                state = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
                returns: 3315 (binary: 110011001100)
            """
            if state is None:
                state = self.agent_state
            
            action = 0
            for i in range(12):
                if state[i]:
                    action |= (1 << i)  # Set bit i
            return action

    def render(self):
        """
        Render traffic intersection with vehicles and traffic light in a specific state
        """
        
        print()
        print(self.agent_state)
        print()
        def light_symbol(cond): 
            return "G " if cond else "R "

        q = self.queue_length
        width = q * 2 + 9

        # --- TOP SECTION ---
        for i in range(q + 1):
            for j in range(width):
                if i < q:
                    if j <= q:
                        val = self.top_left[i, j]
                    elif j <= q + 3:
                        val = self.north_after_junc[i, j - q - 1]
                    elif j == q + 4:
                        val = " "
                    elif j <= q + 7:
                        val = self.south_before_junc[i, j - q - 5]
                    else:
                        val = self.top_right[i, j - q - 8]
                elif i == q:
                    if j <= q:
                        val = self.top_left[i, j]
                    elif j <= q + 4:
                        val = " "
                    elif q + 5 <= j <= q + 7:
                        val = light_symbol(self.agent_state[j - q - 2])
                    else:
                        val = self.top_right[i, j - q - 8]
                print(f"{val:<2}", end="")
            print()

        # --- MIDDLE SECTION ---
        for i in range(7):
            for j in range(width):
                val = " "
                if i < 3:
                    if j < q:
                        val = self.east_before_junc[i, j]
                    elif j == q:
                        val = light_symbol(self.agent_state[i + 6])
                    elif q < j < q + 8:
                        if j not in (q + 4, q + 8):
                            offset = j - q - (1 if j < q + 4 else 2)
                            val = self.junction[i, offset]
                    elif j >= q + 9:
                        val = self.east_after_junc[i, j - q - 9]
                elif i == 3:
                    val = " "
                else:  # i > 3
                    if j < q:
                        val = self.west_after_junc[i - 4, j]
                    elif q < j < q + 8:
                        if j not in (q + 4, q + 8):
                            offset = j - q - (1 if j < q + 4 else 2)
                            val = self.junction[i - 1, offset]
                    elif j == q + 8:
                        val = light_symbol(self.agent_state[9 - i])
                    elif j > q + 8:
                        val = self.west_before_junc[i - 4, j - q - 9]
                print(f"{val:<2}", end="")
            print()

        # --- BOTTOM SECTION ---
        for i in range(q + 1):
            for j in range(width):
                if i > 0:
                    if j <= q:
                        val = self.btm_left[i, j]
                    elif j <= q + 3:
                        val = self.north_before_junc[i - 1, j - q - 1]
                    elif j == q + 4:
                        val = " "
                    elif j <= q + 7:
                        val = self.south_after_junc[i - 1, j - q - 5]
                    else:
                        val = self.btm_right[i, j - q - 8]
                else:
                    if j <= q:
                        val = self.btm_left[0, j]
                    elif q < j <= q + 3:
                        val = light_symbol(self.agent_state[j - q - 1])
                    elif j <= q + 7:
                        val = " "
                    else:
                        val = self.btm_right[0, j - q - 8]
                print(f"{val:<2}", end="")
            print()
        print()
        print("=" * 60)
        print()               

# Example Usage and Demonstrations
if __name__ == "__main__":
    # Testing of Env
    traffic_light_status = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1] #define which traffic light to trigger
    queue_length = 1
    occupancy = np.zeros((4, 3, queue_length), dtype=int)
    occupancy[0, 1, 0] = 1  # Set the value to match your previous example

    action = 4095 #switch from binary to int for input
    #initialise inputs needed for agent
    traffic_rate_upstream = [1,1,1,1] #spawn rate set to 100% for all direction
    traffic_rate_downstream = [1,1,1,1] #clearing rate set to 100% for all direction

    Env_input = TrafficLightRewards(action, occupancy ,queue_length, traffic_rate_upstream, traffic_rate_downstream)
    env = demo_env(Env_input, queue_length)
    

    for i in range(3):  # apply action
        env.step(action)
        env.render()

