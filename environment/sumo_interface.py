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

TODO - Add downstream() method for controlling speed
TODO - Add number of cars in intersection
"""
import os
import sys
import time
import heapq
import argparse
from itertools import permutations

from utils import *

import numpy as np

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

# SUMO Names
X = "A0"
N, E, S, W = "top0", "right0", "bottom0", "left0"
ROADS = [N, E, S, W]
ROUTES = np.array(list(permutations(range(4), 2)))

### === Utilities ===
def rand_routes(n=1, p=None):
    """Select n random routes."""
    selection = np.random.choice(list(range(len(ROUTES))), size=n, replace=False, p=p)
    return ROUTES[selection]

### === For Ease Of Use ===
def readable_road_name(road):
    return ('N', 'E', 'S', 'W')[road_index(road)]

def route_name(a, b=None):
    """Get a name for route from road a through intersection to b."""
    if b is not None:
        return f"{readable_road_name(road_index(a))}{readable_road_name(road_index(b))}"
    return road_index(a[0]), road_index(a[1])

def lane_name(lane, *, out=False):
    if isinstance(lane, str):
        if lane[0] == ":":
            return -1
        lane_index = int(lane.split("_")[-1])
        road0, road1, *_ = map(lambda r: "{}0".format(r), lane.split("0"))
        if road0 == X:
            return py_index(ROADS, road1) + lane_index
        else:
            return py_index(ROADS, road0) + lane_index
    if lane in range(12):
        road = lane // 3
        lane = lane % 3
        if out:
            return "{}{}_{}".format(X, ROADS[road], str(lane))
        else:
            return "{}{}_{}".format(ROADS[road], X, str(lane))
    else:
        notify_error(ValueError, "lane_name", "'lane' is out of range:", lane)


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
    def __del__(self):
        self._sim.close()

    def get_in_intersection(self):
        return self._in_section
    
    def get_left_intersection(self):
        return self._left_section

    def _update_cars(self):
        for v in self._sim.simulation.getArrivedIDList():
            _ = self._cars.pop(v, None)

        for v in self._sim.simulation.getDepartedIDList():
            start, end = route_name(self._sim.vehicle.getRouteID(v))
            self._cars[v] = [None, None, start, end]

        # self._last = self._in_section
        self._in_section = np.zeros(4)
        self._left_section = np.zeros(4)
        for v in self._sim.vehicle.getIDList():
            lane = self._sim.vehicle.getLaneID(v)
            route = self._sim.vehicle.getRouteID(v)
            start, _ = route_name(route)
            if lane[0] == ":":
                self._in_section[start] = self._in_section[start] + 1
                lane = -1
                lane_pos = None
            elif lane[0:len(X)] == X:
                if self._cars[v][0] != -2:
                    self._left_section[start] = self._left_section[start] + 1
                lane = -2
                lane_pos = None
            else:
                lane = lane_name(lane)
                lane_pos = self._sim.vehicle.getLanePosition(v)
                lane_pos = 1 - lane_pos / self._lengths[lane]
            self._cars[v][0] = lane
            self._cars[v][1] = lane_pos
        # self._exited = self._last_inter - self._inter - self._entered

    def _update_lengths(self):
        for i in range(12):
            self._lengths[i] = self._sim.lane.getLength(lane_name(i))

    def __init__(self, fname, *, fdir=None, gui=False, cfg=None, uid=0, sil=True):
        ## TODO - Explore making it faster by registering subscriptions
        ##        test usign sil=False
        # cfg = cfg or {}

        # Keep track of car positions
        # (NodeStart, NodeEnd, Lane(-1 for in intersection), Pos)
        self._cars    = {}
        self._lengths = np.ones(12)
        self._inter = 0
        self._entered = 0
        self._exited = 0
        self._in_section = np.zeros(4)
        self._left_section = np.zeros(4)

        # Extract various config stuff
        self._node  = X
        self._roads = [N, E, S, W]
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
        # repo  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # fdir  = fdir or os.path.join(repo, "sumo-networks")
        # fname = fname if ".sumocfg" in fname else f"{fname}.sumocfg"
        # fpath = os.path.join(fdir, fname)
        fpath = cfg_file(fname, ".sumocfg", fdir=fdir)
        uid   = f"sim{uid}" if isinstance(uid, int) else uid
        exe   = "sumo-gui" if gui else "sumo"
        flgs  = ["--no-step-log", "--no-warnings"] if sil else []

        # Start simulation
        print("Starting sim file", blue(fname), "with path", dim(fpath))
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
        self._update_lengths()
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
        self._update_cars()

    ### === Get State ===
    def get_lights(self):
        return self._lights.copy()

    def get_occupied(self):
        return self._sensor.copy()
    
    def get_occupied_time(self):
        return self._sensor_t.copy()

    def add_cars(self, routes):
        for a, b in routes:
            self.add_car(a, b)

    def add_car(self, a, b):
        car_id = f"car{self._cars_added}"
        self._sim.vehicle.add(car_id, route_name(a, b))
        ## TODO - Verify it was added...
        self._cars_added += 1

    def set_lights(self, lights):
        assert len(lights) == 12
        self._new_lights = np.array(lights)

    def visualize(self):
        """Display occupancy AND lights."""
        lights = self.get_lights()
        occ = self.get_occupied()
        print(blue("Lights:"))
        for i in range(12):
            # Print occupied lanes
            for o in occ[i]:
                print(colbg(" ", 'y' if o else None), end="|")
            # Print traffic light
            l = lights[i]
            print(colbg(" ", 'g' if l else 'r'), end="")
            if i % 3 == 1:
                print(" ", dim(readable_road_name(i // 3)))
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

    tl = SumoInterface("demo", gui=args.gui, cfg={"directions": ["top0", "right0", "bottom0", "left0"]})
    
    for i in range(50):
        pts = np.random.choice(range(int(4)), size=2, replace=False).astype(int)
        tl.add_car(*pts)
        tl.set_lights([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0])
        tl.step()
        # print(f"step {i}")
        # lights = tl.get_occupied()
        inter_arr = tl.get_in_intersection()
        inter = np.sum(inter_arr)
        if inter > 0:
            print(dim(i), "There are", blue(inter), "cars in intersection!")
        elif i % 5 == 0:
            print(dim(i), "Count is 0")
        # print("There are", blue)
        # print(lights.reshape(4,3))
        # print(tl.get_occupied_time())
        # print(f"step time {tl.get_time()}")
        # if i % 5 == 0:
            # tl.visualize()
        # print(tl._new_lights)
        time.sleep(0.1)

    del tl