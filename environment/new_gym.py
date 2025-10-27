from sumo_interface import SumoInterface

import numpy as np
from itertools import accumulate
from operator import or_

VOLUMES  = {"none": 3, "high": 3, "medium": 2, "low": 1}
SLOWDOWN = [1, 0.8, 0.4, 0.1, 0]
USEFUL_ACTIONS = [
       # 0,  # All Red (Transition) # Not used anymore
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

### === GYM ENVIRONMENT === --------------------------------------------
class Gym:
    def __init__(self, duration=3600, *, gui=False, seed=None, jam_prob=0.9, min_jam=2, only_useful=True, ustream=None, sensors=5, drop_lanes=None):
        """Args
        duration <float> seconds in simulation to run
        gui      <bool>  only use when testing a trained agent/pre-programmed logic
        seed     <int>   seed to use for repeatability
        jam_prob <float> probability of slowdown
        only_useful
        """ 
        ## Settings
        self._fname = "map_1"
        self._gui   = gui
        self._endtime = duration
        self._drop_lanes = drop_lanes
        self._seed = seed
        self._dstream_prob = jam_prob
        self._dstream_min  = min_jam
        self._actions = USEFUL_ACTIONS if only_useful else list(range(4096))
        self._volume = VOLUMES[str(ustream).lower()]
        self._nsensors = sensors

        # Sanity checks
        if sensors > 5:
            raise ValueError("Can't have more than 5 sensors!")
        if not (0 <= jam_prob and jam_prob < 1):
            raise ValueError("Can't have probability outside [0, 1)")
        if min_jam not in range(len(SLOWDOWN)):
            raise ValueError("Min jam can't be outside [0,{})".format(len(SLOWDOWN)))
        
        # Create first instance
        self.reset()

    ### === MAIN METHODS === -------------------------------------------
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
        self._stepcount         = 0
        self._queue_penalty     = 0
        self._wait_penalty      = 0
        self._long_wait_penalty = 0
        self._last_step_t       = 0

    def step(self, action):
        if self.done:
            return None
        self._prev_queue_len = self.queue_length

        self._stepcount += 1
        lights = self._unpack(action)
        self.sumo.set_lights(lights)
        self._last_step_t = self.sumo.step()
        self._update_rewards()

        return self.state, self.reward, self.done, self.step_count, self.state_shape, self.delta_queue_length #, self.penalty_wait
        
    ### === HELPERS === ------------------------------------------------
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
        return (self.collisions > 0) or (self.simtime >= self._endtime)

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
        # Calculate reward
        reward = - self._queue_penalty - self._wait_penalty - self._long_wait_penalty
        # Scale by last step duration, otherwise it might cheat
        # by changing a green to red everytime to incur the 6 second
        # delay, effectively shortening the episode
        return reward * self._last_step_t

    ### === INTERNAL === -----------------------------------------------
    def _set_dstream(self):
        """Get downstream slowdown amounts
          REQUIRES
        self._dstream_prob -> Probability of setting any slowdown
        self._rng          -> Random number generator with seed
          SETS
        self._downstream
        sumo slowdown
        """
        ### NOTE - Previous implementation of downstream is wrong
        ### TODO - Fix previous
        self._slowdown   = np.zeros(4, dtype=float)
        self._downstream = np.zeros(4, dtype=int)
        start_jam = self._dstream_min
        end_jam   = len(SLOWDOWN)
        lanes = self._rng.choice(4, size=2, replace=False)
        probs = self._rng.random(2)
        slow  = self._rng.random(2)
        for l, p, s in zip(lanes, probs, slow):
            if self._dstream_prob > p:
                l = int(l)
                slowdown = start_jam + int(np.floor(s * (end_jam - start_jam)))
                self._slowdown[l] = SLOWDOWN[slowdown]
                self._downstream[l] = slowdown
        self.sumo.set_speed_slowdown(self._slowdown)

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

    def _update_rewards(self):
        # To scale them - queue_penalty can be up to 25 per lane
        # Max here is 25 * 12 * 0.05 == 15 -> All lanes entirely full
        # Per car is 0.05 for waiting any duration
        self._queue_penalty     = np.sum(self.queue_length) * 0.05
        
        # Wait penalty can be up to 5 per lane - increase a small amount
        # Max here is 5 * 12 * 0.1 == 6    -> All lanes at least 5 cars
        # Per car is 0.1 for waiting at least 1 second
        self._wait_penalty      = np.sum(self.occupied_time > 5) * 0.05
        
        # Long wait penalty can be up to 5 per lane
        # Max here is 5 * 12 * 0.5 == 30   -> All lanes at least 5 cars long wait
        # Per car is 0.5 * 6 == 6 for waiting at least 1 minute
        self._long_wait_penalty = np.sum(self.occupied_time > 60) * 0.5
        # Max possible reward
        
        

if __name__ == "__main__":
    """NOTE
    In order to "cheat" and run by number of episodes instead of duration,
    set `duration=float('inf')` in the `Gym` class, or `--duration=inf`
    from terminal
    """
    from time import sleep, time
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-g", "--gui",        help="Display SUMO GUI for testing", action="store_true")
    parser.add_argument("-p", "--print",      help="Display some info per iteration", action="store_true")
    parser.add_argument("-r", "--repeats",    help="How many repetitions to run", type=int, default=1)
    parser.add_argument("-d", "--duration",   help="How many seconds to simulate", type=float, default=100)
    parser.add_argument("-s", "--seed",       help="Seed for repeatability", type=int)
    parser.add_argument("-u", "--unchanging", help="Keep same light status every step", action="store_true")
    args = parser.parse_args()


    net_rewards = []
    run_times   = []
    steps       = []
    env = Gym(args.duration, gui=args.gui, seed=args.seed)
    for _ in range(args.repeats):
        net = 0
        start = time()
        env.reset()
        next = np.random.choice(env.actions)
        while not env.done:
            if not args.unchanging:
                next = np.random.choice(env.actions)
            state, reward, done, *_ = env.step(next)
            net += reward
            if args.print:
                print("Reward:", reward)
                print(env.queue_length)
                print(env._wait_penalty)
                print(env._long_wait_penalty)
            if args.print or args.gui:
                sleep(0.2)
        net_rewards.append(net)
        run_times.append(time() - start)
        steps.append(env.step_count)

    print("Average reward over", args.repeats, "episodes:", sum(net_rewards) / len(net_rewards))
    print("Average runtime over", args.repeats, "episodes (seconds):", sum(run_times) / len(run_times))
    print("Average number of steps:", sum(steps) / len(steps))
    
    # Basic test of 10 episodes for 3600 seconds:

    # Test 1: Random policy
    # Avg. reward of -46030 -> -12.7 per second or -74 per step
    # Avg. runtime of 9.8
    # Avg. number of steps was 618 (Almost always changing per step)  

    # Test 2: Unchanging (same every step, still random initial state) policy
    # Avg. reward of -114299 -> -31.75 per second or -31.7 per step
    # Avg. runtime of 13.5 seconds (likely SUMO slowdown due to internal stuff related to car spawn)
    # Avg. number of steps was 3595 (1 state change red-> initial accounts for it not being 3600)