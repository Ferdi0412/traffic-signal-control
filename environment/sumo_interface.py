"""Provides class SumoInterface.

  Example
sim = SumoInterface("demo", gui=True) # Use gui=True for the built-in GUI
sim.add_car(0, 2) # N to S
sim.add_car(1, 2) # E to S
sim.set_lights([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0])
sim.step()
sim.visualize() # For 

  Main Methods
step() returns <None>
    Step the simulation, processing traffic lights and so forth

get_lights() returns <np.array (12x1)>
    Whether the traffic lights are (as of end of last step) green

get_occupied() returns <np.array (12xn)>
    Returns the occupancy state
    NOTE This is currently only n == 1, but that is due to difficulties with xml files, not this method

get_occupied_time() returns <np.array (12xn)>
    Returns the time this sensor has been occupied

add_car(a <int OR str>, b <int OR str>) returns <None>
    Add a car at end of road a, to travel to road b

get_collisions() returns <int>
    Check how many collisions occured
"""
import os
import sys
import time
import heapq
import argparse
from itertools import permutations

import numpy as np

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

# Defaults for sim
X = "A0"
N, E, S, W = "N0", "E0", "S0", "W0"

# For readability
ROADS = {"N": 0, "North": 0, "East": 1, "E": 1,
         "South": 2, "S": 2, "West": 2, "W": 2}
NAMES = ("N", "E", "S", "W")
LEFT_DRIVE = False

# For pretty styling
COLORS = {'r': 31, 'red': 31, 'g': 32, 'green': 32,
          'y': 33, 'yellow': 33, 'b': 34, 'blue': 34}

### === For Printing With Colour ===
def pwarn(msg):
    """Apply yellow bold font to message."""
    return "\033[1m\033[33m" + str(msg) + "\033[0m"

def palarm(msg):
    """Apply red bold font to message."""
    return "\033[1m\033[31m" + str(msg) + "\033[0m"

def pbold(msg):
    """Apply blue bold font to message."""
    return "\033[1m\033[34m" + str(msg) + "\033[0m"

def pgood(msg):
    """Apply green bold font to message."""
    return "\033[1m\033[32m" + str(msg) + "\033[0m"

def pdim(msg):
    """Apply dim font to message."""
    return "\033[2m" + str(msg) + "\033[0m"

def puline(msg):
    """Apply underline font to message."""
    return "\033[4m" + str(msg) + "\033[0m"

def ptxt(msg, col):
    """Apply a text (forgeround) colour to message, and bold."""
    if col is None:
        return msg
    return f"\033[{COLORS[col]}m" + str(msg) + "\033[0m"

def pbg(msg, col):
    """Apply a background to the message."""
    if col is None:
        return msg
    return f"\033[{COLORS[col]+10}m" + str(msg) + "\033[0m"

### === Utilities ===
def py_index(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return None

### === For Ease Of Use ===
def road_id(name):
    """Translate a name, like 'N' or 'North' to id (0 for 'N')."""
    if isinstance(name, (int, np.integer)):
        if -1 < name < 4:
            return name
        raise ValueError(f"Invalid id {pwarn(name)}")
    try:
        return ROADS[name]
    except KeyError:
        raise ValueError(f"Unrecognized road name {pwarn(name)}")

def lane_id(r, l):
    """Get the index of a road and lane."""
    r = road_id(r)
    if isinstance(l, str):
        if l == 'l' or l == 'left':
            l = 0
        elif l == 'f' or l == 'fwd' or l == 'forward':
            l = 1
        elif l == 'r' or l == 'right':
            l = 2
        else:
            raise ValueError(f"Unrecognized lane name {pwarn(l)}")
    return r * 3 + l

def road_name(no):
    """Translate the id/number to name, eg. 0 becomes 'North'."""
    return NAMES[road_id(no)]

def route_name(a, b):
    """Get a name for route from road a through intersection to b."""
    return f"{road_name(road_id(a))}{road_name(road_id(b))}"

def pack_lights(state, n_conns):
    """Returns the 'string' representation for SUMO."""
    raw = ""
    for i in range(12):
        if n_conns[i] == 0:
            continue
        raw += ('g' if state[i] else 'r') * int(n_conns[i])
    return raw

def unpack_lights(raw, n_conns, dummy_vals=None):
    """Returns the 0/1 'state' representation from SUMO string."""
    # Just "sampling" the first of each lane's connections
    idx = 0
    state = np.zeros(12)
    for i in range(12):
        if n_conns[i] == 0:
            if dummy_vals:
                state[i] = dummy_vals[i]
            continue
        state[i] = raw[idx] in ('G', 'g')
        idx += int(n_conns[i])
    return state

### === Main Class ===
class SumoInterface:
    """Main Interface For Sumo."""
    def __init__(self, fname, *, fdir=None, gui=False, cfg=None, uid=0, sil=True):
        ## TODO - Explore making it faster by registering subscriptions
        ##        test usign sil=False
        cfg = cfg or {}
        
        # Extract various config stuff
        self._node  = cfg.get("center", X)
        self._roads = cfg.get("directions", [N, E, S, W])
        self._road_in_name  = [f"{r}{self._node}" for r in self._roads]
        self._road_out_name = [f"{self._node}{r}" for r in self._roads]
        assert len(self._roads) == 4

        # List names of each lane - assuming they exist
        self._lane_name_in, self._lane_name_out = [], []
        for n in self._roads:
            for i in range(3):
                self._lane_name_in.append(f"{n}{self._node}_{i}")
                self._lane_name_out.append(f"{self._node}{n}_{i}")
        
        # Get fullpath to .net.xml and .sumocfg and setup sim
        repo  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fdir  = fdir or os.path.join(repo, "sumo-networks")
        fname = fname if ".sumocfg" in fname else f"{fname}.sumocfg"
        fpath = os.path.join(fdir, fname)
        uid   = f"sim{uid}" if isinstance(uid, int) else uid
        exe   = "sumo-gui" if gui else "sumo"
        flgs  = ["--no-step-log", "--no-warnings"] if sil else []

        # Start simulation
        print("Starting sim file", pbold(fname), "with path", pdim(fpath))
        traci.start([exe, "-c", fpath, *flgs], label=uid)
        self._sim = traci.getConnection(uid)

        # Populate various fields
        self._discover_lanes()
        self._discover_routes()
        self._discover_sensors()
        self._initialize() # Everything retrieved from sim

    ### === Startup Methods ===
    def _discover_lanes(self):
        self._lanes_in, self._lanes_out = np.zeros(12), np.zeros(12)
        for l in self._sim.lane.getIDList():
            if l[0] == ":":
                continue # Don't worry about internal lanes right now
            iin  = py_index(self._lane_name_in, l)
            if iin is not None:
                self._lanes_in[iin] = 1
            iout = py_index(self._lane_name_out, l)
            if iout is not None:
                self._lanes_out[iout] = 1
        self._lane_conns = np.zeros(12)
        look_for = self._node
        for i, n in enumerate(self._lane_name_in):
            if not self._lanes_in[i]:
                continue
            for link in self._sim.lane.getLinks(n):
                to, via, *_ = link
                if to[:len(look_for)] == look_for:
                    # if True: #via[0] != ":":
                    self._lane_conns[i] += 1

    def _discover_routes(self):
        # "Lock" from discovering routes twice...
        try:
            self._routes_discovered
            return
        except AttributeError:
            self._routes_discovered = True
        # Add all routes for all connected roads
        for a, b in permutations(range(len(self._roads)), 2):
            if self._lanes_in[a] and self._lanes_out[b]:
                edges = (self._road_in_name[a], self._road_out_name[b])
                self._sim.route.add(route_name(a, b), edges)
    
    def _discover_sensors(self):
        # "Lock" from discovering routes twice...
        try:
            self._sensors_discovered
            return
        except AttributeError:
            self._sensors_discovered = True
        # Keep a priority queue for each lane
        self._sensor_names = [None] * 12
        sensors = [[] for _ in range(12)]
        for loop_id in self._sim.inductionloop.getIDList():
            lane = self._sim.inductionloop.getLaneID(loop_id)
            lane_no = py_index(self._lane_name_in, lane)
            if lane_no is None:
                continue
            pos  = self._sim.inductionloop.getPosition(loop_id)
            heapq.heappush(sensors[lane_no], (-pos, loop_id))
        n = min((len(v) or float('inf')) for v in sensors)
        for i, s in enumerate(sensors):
            if len(s) is 0:
                continue
            self._sensor_names[i] = []
            for _ in range(n):
                npos, loop_id = heapq.heappop(s)
                self._sensor_names[i].insert(0, loop_id)

    def _initialize(self):
        self._cars_added = 0
        self._lights, self._new_lights = np.zeros(12), np.zeros(12)
        self._update_lights()
        sensor_len = max(len(s or []) for s in self._sensor_names)
        self._sensor, self._sensor_t = map(np.zeros, [[12, sensor_len]] * 2)
        self._update_sensors()
        self._time = self._sim.simulation.getTime()

    def _update_lights(self):
        raw = self._sim.trafficlight.getRedYellowGreenState(self._node)
        self._lights = unpack_lights(raw, self._lane_conns, self._new_lights)
        self._new_lights = self._lights.copy()

    def _update_sensors(self):
        for i in range(12):
            if self._sensor_names[i] is None:
                continue
            ## TODO - "Vanishing" bug
            for j, n in enumerate(self._sensor_names[i]):
                data = self._sim.inductionloop.getVehicleData(n)
                if len(data) > 0:
                    self._sensor_t[i, j] = self._time - data[0][2]
                    self._sensor[i, j] = 1
                else:
                    self._sensor_t[i, j], self._sensor[i, j] = 0, 0

    ### === Step ===
    def step(self):
        # Only thing to send/set is lights
        raw = pack_lights(self._new_lights, self._lane_conns)
        self._sim.trafficlight.setRedYellowGreenState(self._node, raw)
        # Step simulation
        self._sim.simulationStep()
        # Update important features
        self._time = self._sim.simulation.getTime()
        self._update_lights()
        self._update_sensors()

    ### === Get State ===
    def get_lights(self):
        return self._lights.copy()

    def get_occupied(self):
        return self._sensor.copy()
    
    def get_occupied_time(self):
        return self._sensor_t.copy()

    def add_car(self, a, b):
        car_id = f"car{self._cars_added}"
        self._sim.vehicle.add(car_id, route_name(a, b))
        ## TODO - Verify it was added...
        self._cars_added += 1

    def visualize(self):
        """Display occupancy AND lights."""
        lights = self.get_lights()
        occ = self.get_occupied()
        print(ptxt("Lights:", 'b'))
        for i in range(12):
            # Print occupied lanes
            for o in occ[i]:
                print(pbg(" ", 'y' if o else None), end="|")
            # Print traffic light
            l = lights[i]
            print(pbg(" ", 'g' if l else 'r'), end="")
            if i % 3 == 1:
                print(" ", pdim(road_name(i // 3)))
            else:
                print() # Just newline
    
    def get_time(self):
        return self._time
    
    ### === Some Config Stuff ===
    def get_collisions(self):
        return self._sim.simulation.getCollidingVehiclesNumber()

    def get_step_time(self):
        return self._sim.simulation.getDeltaT()

    def set_step_time(self, delta):
        self._sim.simulation.setDeltaT(delta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gui", action="store_true", help="Use SUMO GUI")
    args = parser.parse_args()

    tl = SumoInterface("demo.sumocfg", gui=args.gui, cfg={"directions": ["top0", "right0", "bottom0", "left0"]})
    
    for i in range(300):
        pts = np.random.choice(range(int(4)), size=2, replace=False).astype(int)
        tl.add_car(*pts)
        tl.step()
        if i % 5 == 0:
            tl.visualize()
        time.sleep(0.1)