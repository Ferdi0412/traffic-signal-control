from sumo_interface import SumoInterface

import numpy as np
from itertools import accumulate
from operator import or_

VOLUMES  = {"none": 5, "high": 5, "medium": 3, "low": 1}
SLOWDOWN = [1, 0.8, 0.4, 0.1, 0]
USEFUL_ACTIONS = [
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
# SL_NAMES = ["stop", "high", "medium", "low", "none"]

class Gym:
    """
      Properties
    .occupied_time      <array 12 x 5>
    .queue_length       <array 12 x 1>
    .delta_queue_length <array 12 x 1>
    .collisions         <int>
    .simtime            <float>
    .done               <bool>
    .step_count         <int>
    .actions            <list of int>
    .valid_actions      <list of int>
    """
    def _set_dstream(self):
        """Get downstream slowdown amounts
          REQUIRES
        self._dstream_prob -> Probability of setting any slowdown
        self._rng          -> Random number generator with seed
          SETS
        self._downstream
        sumo slowdown
        """
        self._slowdown   = np.zeros(4, dtype=float)
        self._downstream = np.zeros(4, dtype=int)
        lanes = self._rng.choice(4, size=2, replace=False)
        probs = self._rng.random(2)
        slow  = self._rng.random(2)
        for l, p, s in zip(lanes, probs, slow):
            if self._dstream_prob > p:
                l = int(l)
                slowdown = int(np.floor(s * len(SLOWDOWN)))
                self._slowdown[l] = SLOWDOWN[slowdown]
                self._downstream[l] = slowdown
        self.sumo.set_speed_slowdown(self._slowdown)


        # # Set downstream traffic flows
        # ### NOTE - This is incorrectly handled in the original TrafficGym
        # ### TODO - Rectify this across variants
        # if isinstance(dstream, int):
        #     index = dstream
        #     dstream = np.zeros(4, dtype='U10')
        #     for i in range(4):
        #         dstream[i] = "high" if i == index else "none"
        # if dstream:
        #     if len(dstream) != 4:
        #         raise ValueError("Malformed dstream", dstream)
        #     self.dstream = np.zeros(4, dtype=float)
        #     for i, d in enumerate(dstream):
        #         self.dstream[i] = SLOWDOWN[str(d).lower()]
        # else:
        #     self.dstream = np.zeros(4, dtype=float)
        # sim.set_speed_slowdown(self.dstream)


    def _set_ustream(self):
        """Get upstream traffic volumes
          REQUIRES
        self._drop_lanes -> True for lanes that shouldn't exist
        self._volume     -> Cars / second
        self._rng        -> Random number generator w/ appropriate seed
          SETS
        self._upstream
        sumo prob
        """
        # Compute traffic volumes per lane
        ### NOTE - Adding the ability to disable some lanes
        ###        in order to emulate roads without 3 lanes per dir.
        ###        or which are missing a direction entirely
        _drop_lanes = self._drop_lanes
        if self._drop_lanes:
            _drop_lanes = np.array(_drop_lanes, dtype=bool)
            if _drop_lanes.shape != (12,):
                raise ValueError("droplanes expected shape (12,), got", _drop_lanes.shape)
            count   = 12 - np.sum(_drop_lanes)
            if count < 7:
                raise ValueError("Can't drop more than 5 lanes!")
            self._traffic = np.ones(12, dtype=float) * (self._volume / count)
            self._traffic[_drop_lanes] = 0
        else:
            self._traffic = np.ones(12, dtype=float) * (self._volume / 12)
        ustream = 0 if self._volume == 1 else 1 if self._volume == 3 else 5
        self._upstream = np.ones(4, dtype=int) * ustream
        self.sumo.set_car_prob(self._traffic)

    @staticmethod
    def _unpack(action_value):
        """Translate 0 to [0] * 12 and 4095 to [1] * 12"""
        return [((action_value >> i) & 1) for i in range(12)]

    @staticmethod
    def _pack(light_list):
        """Translate [0] * 12 to 0 and [1] * 12 to 4095"""
        return sum(l << i for i, l in enumerate(light_list))

    def __init__(self, duration=3600, *, gui=False, seed=None, jam_prob=0.7, only_useful=True, ustream=None, sensors=5, drop_lanes=None):
        """Args
        simcfg <dict> -> SumoInterface kwargs
            [default: {"fname": "map_1"}]
        maxtime <float> -> Maximum simulation time for episode
            [default: 3600]
        ustream <str> -> Value from VOLUMES (above) representing average
                         number of cars per second into intersection
            [default: 5]
        droplanes <array 12: bool> -> Set `True` for up to 5 lanes to drop
            [default: [False] * 12]
        dstream 
            VARIANT 1: <array 4: str> -> Value from SLOWDOWN (above)
                                         representing how much a car slows down
                                         on it's way into intersection (1 full
                                         full stop, 0 no slowdown)
            VARIANT 2: <int> -> Index of lane to bring to "high" slowdown
            [default: ["none"] * 4]
        nsensors <int max 5> -> Number of sensors per lane to return
                                The shape of sensor array is always
                                (12, 5), but this will right-pad w/ 0s
            [default: 5]
        """ 
        ## Settings
        self._fname = "map_1"
        self._gui   = gui
        self._endtime = duration
        self._drop_lanes = drop_lanes
        self._seed = seed
        self._dstream_prob = jam_prob
        self._actions = USEFUL_ACTIONS if only_useful else list(range(4096))
        self._volume = VOLUMES[str(ustream).lower()]
        self._nsensors = sensors

        # Sanity checks
        if sensors > 5:
            raise ValueError("Can't have more than 5 sensors!")
        if not (0 <= jam_prob and jam_prob < 1):
            raise ValueError("Can't have probability outside [0, 1)")
        
        # Create first instance
        self.reset()

    def reset(self):
        """Create a fresh simulation."""
        self._rng = np.random.default_rng(self._seed)
        
        simcfg = {
            "fname": self._fname,
            "seed":  self._seed,
            "gui":   self._gui,
        }
        self.sumo = SumoInterface(**simcfg)
        self.sumo.set_lights([0] * 12)
        self._set_ustream()
        self._set_dstream()

        # Internal values
        self._prev_queue_len = np.zeros(12, dtype=int)
        self._stepcount = 0
        self._queue_penalty     = 0
        self._wait_penalty      = 0
        self._long_wait_penalty = 0

    @property
    def occupied_time(self):
        if self._nsensors == 5:
            return self.sumo.get_occupied_time()
        else:
            return self.sumo.get_occupied_time()[:, :self._nsensors]
    
    @property
    def queue_length(self):
        return self.sumo.get_queue_length()

    @property
    def delta_queue_length(self):
        return self.queue_length - self._prev_queue_len

    @property
    def collisions(self):
        return self.sumo.get_collisions()
    
    @property
    def simtime(self):
        return self.sumo.get_time()
    
    @property
    def done(self):
        self.collisions > 0 or self.simtime >= self._endtime

    @property
    def step_count(self):
        return self._stepcount

    @property
    def actions(self):
        return self._actions

    @property
    def valid_actions(self):
        return self.actions

    @property
    def downstream_status(self):
        return self._downstreams

    @property
    def upstream_status(self):
        return self._upstream

    @property
    def traffic_light(self):
        return self.sumo.get_lights()

    @property
    def state(self):
        return self.traffic_light, self.occupied_time, self.queue_length

    @property
    def state_shape(self):
        return self.traffic_light.shape, self.occupied_time.shape, self.queue_length.shape

    @property
    def reward(self):
        return self._queue_penalty + self._wait_penalty + self._long_wait_penalty

    def step(self, action):
        if self.done:
            return None
        self._prev_queue_len = self.queue_length

        self._stepcount += 1
        lights = self._unpack(action)
        self.sumo.set_lights(lights)
        self.sumo.step()
        self._update_rewards()

        return self.state, self.reward, self.done, self.step_count, self.state_shape, self.delta_queue_length #, self.penalty_wait
        
    def _update_rewards(self):
        self._queue_penalty     = np.sum(self.queue_length)
        self._wait_penalty      = np.sum(np.where(self.occupied_time > 5))
        self._long_wait_penalty = np.sum(np.where(self.occupied_time > 60)) * 5
        return self._queue_penalty + self._wait_penalty + self._long_wait_penalty
        
        

if __name__ == "__main__":
    env = Gym(gui=True)

    for step in range(100):
        env.step(0)

    env.reset()