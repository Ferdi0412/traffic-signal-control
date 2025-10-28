"""Provides class "SumoInterface".
   Took 330s for 100 episodes of 1000 steps
   Previous variant was over twice as slow, at around 700s for similar

  | Example:
from sumo_interface import SumoInterface

a = [0] * 3 + [1] * 3 + [0] * 3 + [1] * 3
b = [1] * 3 + [0] * 3 + [1] * 3 + [0] * 3

sim = SumoInterface("map_1", seed=0, gui=False)
sim.set_car_prob([2 / 12] * 12) # 2 cars every second
sim.set_lights(a) # All off
last_set = sim.get_time()
for _ in range(1000):
if sim.get_time() - last_set > 3:
    next = a if (sim.get_lights() == b).all() else b
    sim.set_lights(next)
sim.step()

  | Methods:
SumoInterface(fname, *, [gui], [seed], ...)
    fname -> Name of file in "{fname}.sumocfg" under "sumo-networks"
    gui   -> If set to 'True' will use SUMO's GUI (slow)

.reset([fname])
    Start a fresh simulation

.step(seconds <float>) -> float
    If no traffic light change, forward by `seconds`
    If traffic light changed, forward `seconds` + 3s (for yellow light)
    Returns the time in seconds it has been running

.get_lights()            -> <12 x 1 array>
.set_lights(lights <12 x 1 array>)
    Get traffic lights, 1 for Green, 0 for Red

.add_car(start, end, [check])          -> <bool>
    Where to start and where to end, in range [0, 4). Pend parameter
    checks whether a car can be spawned there, or if that would clash
    with another recently spawned car.
    NOTE - start & end must differ
    NOTE - Use either this or .set_car_prob - don't use both in same simulation

.set_car_prob(probs <12 x 1 array>)
    The probability to add a car in each lane/direction per second
    (max 1)
        0: N->turn left
        1: N->go straight
        ...
    NOTE - Use either this or .add_car - don't use both in same simulation

.get_time()
    How long the simulation has been run.

.get_occupied()          -> <12 x 5 array>
.get_occupied_time()     -> <12 x 5 array>
    Occupancy of the induction loops. Time is how long current car has
    been there. Portion is what fraction [0, 1] of the last step it was
    occupied.

.get_queue_length()      -> <12 x 1 array>
    SUMO defines the queue length as the number of cars that are
    waiting (moving at speed <0.1 m/s)

.get_in_intersection()   -> <4 x 1 array>
.get_left_intersection() -> <4 x 1 array>
    Whether there are cars in the intersection or not.

.get_speed_slowdown()    -> <4 x 1 array>
.set_speed_slowdown(slowdown <4 x 1 array>)
    How much each exit direction is slowed down by traffic. 0 is not at
    all (exit speed is 50 km/h), 1 is full stop (0 km/h).
"""
### Notes to self
###
### No more "convenience" stuff
### For ROADS:
###   0: North
###   1: East
###   2: South
###   3: West
### For LANES (in direction of traffic):
###   0: leftmost
###   1: middle
###   2: rightmost
###
### Nomenclature:
###   road:  value from 0 to 3
###   lane:  value from 0 to 11
###   turn:  value from 0 to 2 (lane belonging to a road)
###   route: non-equal pair of values in [0, 4)
import os
import sys
import heapq
import math
from itertools import permutations

import numpy as np

# if 'SUMO_HOME' in os.environ:
#     sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
from import_sumo import traci

### === STATIC CONSTANTS === ===========================================
NODE  = "A0"
ROADS = ("top0", "right0", "bottom0", "left0")
NAMES = ("N",    "E",      "S",       "W")
SPEED = 50 / 3.6

### === MAIN CLASS === =================================================
class SumoInterface:
    def __init__(self, fname, *, seed=None, gui=False, sil=True):
        # The following are used to start the SUMO program
        self._file  = cfg_path(fname, None)
        self._cmd   = "sumo-gui" if gui else "sumo"
        self._flags = ["--no-step-log", "--no-warnings"] if sil else []
        # This is used in ._init()
        self._seed = seed
        # Start the SUMO program
        self._start()

    def __del__(self):
        # Close the connection when the class is destroyed
        try:
            self._sim.close()
        except AttributeError:
            pass # Let this one die quietly...
        except Exception as e:
            print(alarm("SumoInterface.__del__"), repr(e))

    def _start(self):
        # Need a new UID for every subsequent simulation
        uid = get_uid()
        traci.start([self._cmd, "-c", self._file, *self._flags], label=uid)
        self._sim = traci.getConnection(uid)
        # Do any additional initialization
        self._init()

    def _init(self):
        self._ytime    = 5
        self._steptime = 1
        self._carindex = 0

        self._steps = 0

        # 1) Get Lanes
        self._linkcounts = np.zeros(12, dtype=int)
        for i in range(12):
            lane = nameof_lane(i, 'in')
            links = self._sim.lane.getLinks(lane)
            self._linkcounts[i] = len(links)

        # 2) Get step time
        self._deltat = self._sim.simulation.getDeltaT()

        # 3) Register routes
        for start, end in permutations(range(4), 2):
            edges = (nameof_road(start, 'in'), nameof_road(end, 'out'))
            name = nameof_route(start, end)
            self._sim.route.add(name, edges)

        # 4) Dummy "left intersection" and cars entering to keep track
        # Approach to track: 
        # Get new cars in
        # Get new cars out
        self._exited   = np.zeros(4, dtype=int)
        self._entered  = np.zeros(4, dtype=int)
        self._inside   = np.zeros(4, dtype=int)

        # 5) No collisions yet
        self._collisions = 0

        # 6) Store names of each induction loop sensor for quick lookup
        sensors = 5
        self._sensornames = np.zeros((12, sensors), dtype='U40')
        self._exitsensornames = np.zeros(12, dtype='U40')
        
        entry = [[] for i in range(12)]
        for loop_id in self._sim.inductionloop.getIDList():
            # We are not using the traffic light's pre-made sensor
            # due to poor positioning
            if loop_id[:3] == "TLS": continue
            lane = self._sim.inductionloop.getLaneID(loop_id)
            i = indexof_lane(lane)
            # Only 1 exit sensor per lane
            if isexit_lane(lane):
                self._exitsensornames[i] = loop_id
                continue
            # Keep priority queue to order closest->farthest
            pos = self._sim.inductionloop.getPosition(loop_id)
            heapq.heappush(entry[i], (-pos, loop_id))
        # Now iterate over sensors and assign correct order
        for i in range(12):
            for s in range(sensors):
                _, loop_id = heapq.heappop(entry[i])
                self._sensornames[i, s] = loop_id

        # 7) Empty arrays to track time and occupancy portion
        self._occupied_t = np.zeros(self._sensornames.shape, dtype=float)
        self._occupied_p = np.zeros(self._occupied_t.shape, dtype=float)

        # 8) Time to avoid AttributeError later
        self._time     = -0.1
        self._prevtime = -0.1

        # 9) Add car probabilities per second
        self._prob  = None
        self._rng   = np.random.default_rng(self._seed)

        # Update values
        self._pre_update()
        self._partial_update()
        self._update()

    def _pre_update(self):
        ### This happens at start of "Interface Step",
        ### before any "Simulation Steps" occur
        self._prevtime = self._time
        self._time = self._sim.simulation.getTime()

        # Keep track of changes here...
        self._entered = np.zeros(4, dtype=int)
        self._exited  = np.zeros(4, dtype=int)
        self._pending = np.zeros(12, dtype=int)

        # Keep track of new aggregate
        self._occupied_p = np.zeros(self._occupied_t.shape, dtype=float)

    def _partial_update(self):
        ### This one happens once per "Simulation Step"
        ### The intent is to keep track of certain aggregate metrics
        ###
        ### If wanted in future -> _occupied_p
        # 0) Time
        self._time = self._sim.simulation.getTime()

        # 1) Collisions
        self._collisions += self._sim.simulation.getCollidingVehiclesNumber()

        # 2) Induction Sensors
        _, nsensors = self._sensornames.shape
        for i in range(12):
            for s in range(nsensors):
                loop_id = self._sensornames[i, s]
                data = self._sim.inductionloop.getVehicleData(loop_id)
                count = len(data)
                # If no cars passed it, set occupied_t to 0
                if count == 0:
                    self._occupied_t[i, s] = 0
                    continue
                _, _, t_enter, t_exit, _ = data[-1]
                fully_occ = False
                if t_exit == -1:
                    t_occ = self._time - t_enter
                    self._occupied_t[i, s] = t_occ
                    fully_occ = t_occ > self._deltat
                    count -= 1 # Car not entered...
                # The occupied_p will be scaled in final ._update() call
                if fully_occ:
                    self._occupied_p[i, s] += 100
                else:
                    portion = self._sim.inductionloop.getLastStepOccupancy(loop_id)
                    self._occupied_p[i, s] += portion
                # If closest to intersection, update entered field
                if s == 0:
                    self._entered[i // 3] += sum(t_exit != -1 for _, _, _, t_exit, _ in data)
        
        # 3) Intersection exit induction sensors
        for i in range(12):
            loop_id = self._exitsensornames[i]
            data = self._sim.inductionloop.getVehicleData(loop_id)
            for (car, _, _, t_exit, _) in data:
                # If car hasn't fully passed yet, skip it
                if t_exit == -1: continue
                route = self._sim.vehicle.getRouteID(car)
                start, _ = endsof_route(route)
                self._exited[start] += 1

        # 4) Add cars probabilistically
        if self._prob is not None:
            r = self._rng.random(12)
            for lane in np.where(self._prob >= self._rng.random(12))[0]:
                self.add_car(lane // 3, lane_turnsto(lane), False)

        # 5) Update pending
        for c in self._sim.simulation.getDepartedIDList():
            route = self._sim.vehicle.getRouteID(c)
            i = indexof_route(*endsof_route(route))
            self._pending[i] -= 1

    def _update(self):
        ### This one happens once per "Interface Step"
        ### The intent is to update items which will be accessed more
        ### than once per iteration "Interface Step"
        
        # 1) Lights
        raw = self._sim.trafficlight.getRedYellowGreenState(NODE)
        self._lights = unpack(raw, self._linkcounts)
        self._next   = None

        # 2) Inside Lane tracking
        self._inside += self._entered - self._exited

        # 4) Scale Occupied Portion by number of "Simulation Steps"
        self._occupied_p /= (self._time - self._prevtime) / self._deltat

    def _newlight(self):
        if self._next is None:
            return None, None
        if (self._next == self._lights).all():
            return None, None
        return pack(self._next, self._linkcounts), pack_yellow(self._lights, self._next, self._linkcounts)

    def _nextcar(self):
        self._carindex += 1
        return "car{}".format(self._carindex)

    ### --- Main Functions --- -----------------------------------------
    def reset(self, fname=None, *, fdir=None):
        """Replaces this sim with a fresh/new simulation."""
        self._sim.close()
        self._start()

    def step(self):
        """Go to the next "environment step"."""
        self._pre_update()
        # Get new traffic light if it has changed
        newlight, ylight = self._newlight()
        if newlight is not None:
            # Wait for appropriate yellow light duration
            self._sim.trafficlight.setRedYellowGreenState(NODE, ylight)
            for _ in range(int(self._ytime / self._deltat)):
                self._sim.simulationStep()
                self._partial_update()
            # Then set new green/red lights
            self._sim.trafficlight.setRedYellowGreenState(NODE, newlight)
        # Step for desired "evironment step duration"
        for _ in range(int(self._steptime / self._deltat)):
            self._sim.simulationStep()
            self._partial_update()
        # Get elapsed time in sim
        self._update()
        self._steps += 1
        return self.get_last_step_time()

    def sim_step_count(self):
        return self._time / self._deltat

    def step_count(self):
        return self._steps

    def get_last_step_time(self):
        return self._time - self._prevtime

    def get_lights(self):
        """Get <array 12 x 1> of 0/1 for red/green light."""
        return self._lights

    def set_lights(self, lights):
        """Set <array 12 x 1> of 0/1 for red/green light."""
        self._next = np.array(lights, dtype=int)

    def add_car(self, start, end, check=True):
        """Add car going from start <int> -> end <int>.
        Start cannot be the same as end.
        [check] <bool> whether to only add cars if they can immediately
                       spawn
        Returns <bool> whether car spawned.
        """
        lane = indexof_route(start, end)
        if check and self._pending[lane]:
            return False
        route = nameof_route(start, end)
        turn  = lane % 3 
        self._sim.vehicle.add(self._nextcar(), route, departLane=turn)
        self._pending[lane] += 1
        return True

    def set_car_prob(self, probs):
        probs = np.array(probs, dtype=float)
        if not ((probs <= 1).all() and (probs > 0).all()):
            raise ValueError("Must set probs in <array 12 x 1> in range [0, 1]!")
        self._prob = probs * self._deltat

    def get_time(self):
        """Retrieve <float> simulation time elapsed."""
        return self._time

    def get_occupied(self):
        """Get <array 12 x 1> 0/1 for if a car is over a sensor."""
        return (self._occupied_t > 0.0).astype(int)

    def get_occupied_time(self):
        """Get <array 12 x 1: float> time current car has been over a sensor."""
        return self._occupied_t

    def get_occupied_portion(self):
        """Get <array 12 x 1: float> from 0 to 100 for % of 
        latest "environment step" the sensor was occupied.
        """
        return self._occupied_p

    def get_queue_length(self):
        """Get <array 12 x 1: int> number of cars not moving."""
        lengths = np.zeros(12, dtype=int)
        for i in range(12):
            lane = nameof_lane(i, 'in')
            lengths[i] = self._sim.lane.getLastStepHaltingNumber(lane)
        return lengths

    def get_in_intersection(self):
        """Get <array 4 x 1: int> number of cars currently passing
        through the intersection.
        """
        return self._inside

    def get_left_intersection(self):
        """Get <array 4 x 1: int> number of cars that exited 
        the intersection in the last step.
        """
        return self._exited
        
    def get_entered_intersection(self):
        """Get <array 4 x 1: int> number of cars that entered
        the intersection in the last step.
        """
        return self._entered

    def get_speed_slowdown(self):
        """Get <array 4 x 1: float> from 0 to 1 for how much
        traffic is slowed down - 1 is full stop, 0 is max speed.
        """
        speeds = np.zeros(4, dtype=float)
        for road in range(4):
            lane = nameof_lane(road * 3, 'out')
            speeds[road] = self._sim.lane.getMaxSpeed(lane)
        return (SPEED - speeds) / SPEED 

    def set_speed_slowdown(self, speeds):
        """Set speed <array 4 x 1: float> from 0 to 1 for how
        much to slow down traffic by - 1 is full stop, 0 is max speed.
        """
        speeds = (1 - np.array(speeds, dtype=float)) * SPEED
        for road, speed in enumerate(speeds):
            for turn in range(3):
                lane = nameof_lane(road * 3 + turn, 'out')
                self._sim.lane.setMaxSpeed(lane, speed)

    def get_collisions(self):
        """Get number of collisions that occured during the entire
        simulation.
        """
        return self._collisions
    
    def get_cars_added(self):
        """Check how many cars have been added to the simulation."""
        return self._carindex

    ### --- TO GET RANGE OF STEPS --------------------------------------
    def min_step_t(self):
        """Note - based off assumption of deltat := 0.1"""
        return self._steptime
    
    def max_step_t(self):
        """Note - based off assumption of deltat := 0.1"""
        return self._ytime + self._steptime

    def steps_needed(self, time):
        """Returns (lower limit, upper limit)"""
        return (math.floor(time / self.max_step_t()), math.ceil(time / self.min_step_t()))

    ### --- FOR TESTING --- --------------------------------------------
    def visualize(self):
        """Get a terminal view of the key "state space"."""
        lights = self.get_lights()
        occ    = self.get_occupied()
        inside = self.get_in_intersection()
        queues = self.get_queue_length()
        for i in range(12):
            l = lights[i]
            print(colbg("o", 'g' if l else 'r'), end="")
            for o in occ[i]:
                print(colbg(" ", 'y' if o else None), end="|")
            if i % 3 == 1:
                road = i // 3
                q = np.sum(queues[road*3:(road+1)*3])
                print("{}>".format(inside[road]),
                      dim(NAMES[road]),
                      comment(q))
            else:
                print() # empty newline

    ### --- FOR DRAWING --- --------------------------------------------
    def get_viewport(self):
        # Output shape is [x, y, x, y]
        (x0, y0), (x1, y1) = self._sim.simulation.getNetBoundary()
        return np.array([x0, y0, x1, y1], dtype=float)

    def get_intersection_shape(self):
        # Output shape is [[x, y], ...]
        return np.array(self._sim.junction.getShape(NODE), dtype=float)

    def get_lane_midpoints(self, direction):
        # Output shape is [[x0, y0, x1, y1], ...]
        """direction must be 'in' or 'out'."""
        midpoints = np.zeros((12, 4), dtype=float)
        for i in range(12):
            name = nameof_lane(i, direction)
            shape = self._sim.lane.getShape(name)
            x0, y0 = shape[0]
            x1, y1 = shape[-1]
            midpoints[i] = [x0, y0, x1, y1]
        return midpoints

    def get_lane_shape(self, direction):
        # Output is [[x0, y0, x1, ..., x3, y3], ...]
        """direction must be 'in' or 'out'."""
        edges = np.zeros((12, 8), dtype=float)
        for i in range(12):
            name = nameof_lane(i, direction)
            shape = self._sim.lane.getShape(name)
            half_width = self._sim.lane.getWidth(name) / 2
            m0 = shape[0]
            m1 = shape[-1]
            x0, y0 = perp(m0, m1, half_width)
            x1, y1 = perp(m0, m1, -half_width)
            x2, y2 = perp(m1, m0, half_width)
            x3, y3 = perp(m1, m0, -half_width)
            edges[i] = x0, y0, x1, y1, x2, y2, x3, y3
        return edges

    def get_car_midpoints(self):
        # x, y, angle, length, width
        car_ids = self._sim.vehicle.getIDList()
        cars = np.zeros((len(car_ids), 5), dtype=float)
        for i, c in enumerate(car_ids):
            x, y = self._sim.vehicle.getPosition(c)
            a = self._sim.vehicle.getAngle(c)
            l = self._sim.vehicle.getLength(c)
            w = self._sim.vehicle.getWidth(c)
            cars[i] = x, y, a, l, w
        return cars

    def get_sensor_positions(self):
        # Output is [([x0, y0], ..., [x3,y3]), ...]
        _, n = self._sensornames.shape
        shapes = np.full((12, n, 2), np.nan, dtype=float)
        lanes = self.get_lane_midpoints('in')
        for i in range(12):
            x0, y0, x1, y1 = lanes[i]
            for j in range(n):
                name = self._sensornames[i, j]
                if len(name) == 0:
                    continue
                pos = self._sim.inductionloop.getPosition(name)
                shapes[i, j] = proj((x0, y0), (x1, y1), pos)
        return shapes

### === HELPER FUNCTIONS === ===========================================
### For when loading file
_uid = 0

def get_uid():
    global _uid
    uid = _uid 
    _uid += 1
    return uid

def cfg_path(fname, fdir=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fdir = fdir or os.path.join(repo_root, "sumo-networks")
    fname = fname.split(".")[0]
    return os.path.join(fdir, "{}.sumocfg".format(fname))

def unpack(raw, link_counts):
    idx = 0
    state = np.zeros(12, dtype=int)
    for i in range(12):
        state[i] = raw[idx] in ("G", 'g')
        idx += int(link_counts[i])
    return state 

def pack(lights, link_counts):
    pack_one = lambda i, l, c: ("r" if not l else "g" if i % 3 == 2 else "G") * c
    return "".join(pack_one(i, l, c) for i, (l, c) in enumerate(zip(lights, link_counts)))

def pack_yellow(prev, next, counts):
    pack_one = lambda i, p, n, c: ("y" if (p and not n) else "r" if not p else "g" if i % 3 == 2 else "G") * c
    return "".join(pack_one(i, p, n, c) for i, (p, n, c) in enumerate(zip(prev, next, counts)))

def alarm(*msg):
    msg = msg or []
    # ..... bold   blink  red .......................................... reset 
    return "\033[1m\033[5m\033[31m" + " ".join([str(m) for m in msg]) + "\033[0m"

def dim(*msg):
    return "\033[2m" + " ".join([str(m) for m in msg]) + "\033[0m"

def blue(*msg):
    return "\033[1m\033[34m" + " ".join([str(m) for m in msg]) + "\033[0m"

def comment(*msg):
    return "\033[2m\033[32m" + " ".join([str(m) for m in msg]) + "\033[0m"

def indexof_route(start, end):
    """Lane in which a car following a certain route will follow into intersection."""
    return start * 3 + turn_needed(start, end)

def turn_needed(start, end):
    turn = end - start + 1
    while turn > 2:
        turn -= 3
    while turn < 0:
        turn += 3
    return turn

def py_index(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return None

def endsof_route(name):
    start, end = name
    return py_index(NAMES, start), py_index(NAMES, end)

def nameof_route(start, end):
    """Name for a route between 2 directions."""
    return NAMES[start] + NAMES[end]

def nameof_lane(lane, direction):
    return "{}_{}".format(nameof_road(lane // 3, direction), lane % 3)

def indexof_lane(name):
    road, turn = name.split("_")
    return indexof_road(road) * 3 + int(turn)

def indexof_road(name):
    if name[:2] == NODE:
        return py_index(ROADS, name[2:])
    return py_index(ROADS, name[:-2])

def isexit_lane(name):
    return name[:2] == NODE

def nameof_road(road, direction):
    if direction == 'out':
        return "{}{}".format(NODE, ROADS[road])
    return "{}{}".format(ROADS[road], NODE)

def lane_turnsto(lane):
    start = lane // 3
    turn  = lane % 3
    end   = start + turn + 1
    if end > 3:
        end %= 4
    return end

COLORS = {'r': 31, 'red': 31, 'g': 32, 'green': 32,
          'y': 33, 'yellow': 33, 'b': 34, 'blue': 34}
def colbg(msg, col):
    """Apply a background to the message."""
    if col is None:
        return msg
    return f"\033[{COLORS[col]+10}m" + msg + "\033[0m"

def proj(p0, p1, d):
    """Project d from p0 towards p1."""
    p0, p1 = map(np.array, (p0, p1))
    dp = (p1 - p0) / np.linalg.norm(p1 - p0)
    res = p0 + d * dp
    return res[0], res[1]

def perp(p0, p1, d=None):
    """Line perpendicular to p0->p1, with lentgh d."""
    p0, p1 = map(np.array, (p0, p1))
    r, a = ctp(p1 - p0)
    return p0 + ptc(d or r, a - 90)

def ctp(p, p1=None):
    """Cartesian -> pseudo-polar."""
    x, y = p if p1 is None else (p, p1)
    return np.sqrt(x**2 + y**2), np.rad2deg(np.arctan2(y, x))

def ptc(p, p1=None):
    """Pseudo-polar -> cartesian."""
    r, a = p if p1 is None else (p, p1)
    return r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a))  


### === TESTING === ====================================================
if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="map_1", help="SUMO file to use")
    parser.add_argument("-g", "--gui", action="store_true", help="Whether to show GUI")
    parser.add_argument("-l", "--length", type=int, default=100, help="Length of episode in steps")
    parser.add_argument("-r", "--reset", action="store_true", help="Reset for 2 'playthroughs'")
    parser.add_argument("-t", "--time", action="store_true", help="Time program")
    parser.add_argument("-s", "--seed", action="store_true", help="Set random seed 0")
    args = parser.parse_args()

    if not args.time:
        sim = SumoInterface(args.file, gui=args.gui, seed=0 if args.seed else None)
        
        for i in range(100):
            if i % 2 == 0:
                sim.add_car(0, 3)
                # sim.add_car(i % 3, 3)
            sim.step()
            # if i % 5:
                # sim.visualize()
                # print(sim.get_in_intersection())
            if i == args.length // 2:
                sim.set_lights([0] * 6 + [1] * 6)
                # print(blue("Car midpoints:"))
                print(comment(sim.get_car_midpoints()))
                sim.set_speed_slowdown([4 / 5] * 4)
                # print(comment(sim.get_speed_slowdown()))
                time.sleep(2)
            if args.gui:
                time.sleep(0.1) # Easier to watch

        if args.reset:
            print(blue("Second time"), comment("--->"), comment("have reset sim!"))
            sim.reset()
            for i in range(args.length):
                if i % 2 == 0:
                    sim.add_car((i % 3) + 1, 0)
                sim.step()
                if i % 10:
                    sim.visualize()
                if args.gui:
                    time.sleep(0.1)
    else:
        elapsed = []
        n_iters = 100 if args.reset else 1
        ep_len  = 1000
        for i in range(n_iters):
            start = time.time()
            sim = SumoInterface("map_1", gui=args.gui, seed=0 if args.seed else None)
            # Just because the GUI startup takes significant time
            if args.gui:
                start = time.time() 
            # Set random cars, once per second
            sim.set_car_prob([1 / 12] * 12)
            for s in range(ep_len):
                # sim.add_car(s % 4, (s + 1) % 4)
                sim.step()
                if s % 100 == 0:
                    sim.set_lights([0] * 3 + [1] * 3 + [0] * 3 + [1] * 3)
                elif s % 100 == 50:
                    sim.set_lights([1] * 3 + [0] * 3 + [1] * 3 + [0] * 3)
            elapsed.append(time.time() - start)
            sim.visualize()
            if i % 10 == 0:
                print(blue("Last iteration took", elapsed[-1], "s"))
                print(dim(ep_len, "steps was about"), comment(sim.get_time(), "s"))
                print(dim("And"), sim.get_cars_added(), dim("cars were added"))
        print(dim("Time taken for"), blue(n_iters), ":=", blue(sum(elapsed)))
        print(dim("Average time:"), blue(sum(elapsed) / len(elapsed)))
        print(comment("For episodes of length", ep_len))
        exit()