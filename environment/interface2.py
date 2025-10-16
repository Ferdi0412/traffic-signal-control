"""SumoInteface class

  Constructor
SumoInterface(fname, *, [fdir], [gui], [cfg], [uid], [sil])

<float> road_length(start)
<float> car_length()

  Initial State Control
<None> insert_car_at(start, end, pos)
<None> set_step_time(d_time)

  Flow Control
<bool> add_car(start, end, [allow_pending])
<12 x 1 array> get_cars_pending([lane])
~~ <None> set_speed_out(road, speed)
~~ <float> get_speed_out(road)
    NOTE speeds needs "Variable Speed Signs"

  Action Control
<None> step([lights])
<None> set_lights(lights)
<12 x 1 array> get_lights()

  Non-Controlled State Retrieval
<float> get_time()
<12 x n array> get_occupied()
<12 x n array> get_occupied_time()
<12 x n array> get_occupied_portion()
<4 x 1 array> get_in_intersection()
<4 x 1 array> get_left_intersection()
<int> get_collisions()

  Visualization Retrieval
~~ <m x 2> shape_intersection()
~~ <m x 4> shape_lanes()
~~ <m x 4> shape_cars(road, turn)
~~ <m x 4> shape_sensor()
    NOTE get done after networks generated

  Rough Testing
For 5 episodes of 3600 steps (1 car every 5th step, no other comps.)
took about 85 seconds (17 seconds per episode)

NOTE - Network MUST HAVE 3 lanes in each direction for now...
TODO - Automate lane (link) counts
"""
# Custom methods
from utils import py_index, colbg, alarm, warn, blue, dim, notify_error
from utils import cfg_file, road_index, lane_index

from itertools import permutations

import numpy as np
import heapq

import os, sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

## === Lane And Direction Logic ========================================
DROP_TL = True # Include Traffic Light induction sensors - NOTE poor positioning
NODE  = "A0"
ROADS = ("top0", "right0", "bottom0", "left0")
NAMES = ("N", "E", "S", "W")
TURNS = {"l": 0, "left": 0, "f": 1, "fwd": 1, "forward": 1, "r": 2, "right": 2}

def turn_needed(start: int, end: int):
    turn = end - start + 1
    while turn > 2:
        turn -= 3
    while turn < 0:
        turn += 3
    return turn

def turn_target(start: int, turn):
    if isinstance(turn, str):
        turn = TURNS[turn]
    end = start + turn + 1
    while end > 3:
        end -= 4
    return end

def road_name(index):
    return NODE if index == -1 else ROADS[index]

def edge_name(start, end):
    return "{}{}".format(road_name(start), road_name(end))

def route_name(start, end):
    return "{}{}".format(NAMES[start], NAMES[end])

def route_targets(name):
    return road_index(name[0]), road_index(name[1])

def route_index(name, end=None): # name is also start
    if end is None:
        return route_index(*name)
    s = py_index(NAMES, name) if isinstance(name, str) else name
    e = py_index(NAMES, end) if isinstance(end, str) else end
    if s is None or e is None:
        notify_error(ValueError, "route_index", "Invalid route", name, end)
    return s * 3 + turn_needed(s, e)

def lane_name(index: int, out=False):
    road = index // 3
    lane = index % 3
    if out:
        return "{}{}_{}".format(NODE, ROADS[road], lane)
    else:
        return "{}{}_{}".format(ROADS[road], NODE, lane)

def lane_index(name: str, check_out=False):
    if name[0] == ":":
        notify_error(ValueError, "lane_index", "Internal lane", comment(name))
    roads, turn = name.split("_")
    road0, road1, _ = map(lambda s: "{}0".format(s), roads.split("0")) 
    if road0 == NODE:
        return True if check_out else py_index(ROADS, road1) * 3 + int(turn)
    else:
        return False if check_out else py_index(ROADS, road0) * 3 + int(turn) 

def car_name(index: int):
    return "car{}".format(index)

def car_index(name: str):
    return int(name[3:])

## === "Encoding" ======================================================
def pack(lights: list, lane_count: list):
    pack = lambda i, l, c: ("r" if not l else "g" if i % 3 == 2 else "G") * c 
    return "".join(pack(i, l, c) for i, (l, c) in enumerate(zip(lights, lane_count)))

def unpack(raw: str, lane_count: list, dummy_vals: np.array):
    idx = 0
    state      = np.zeros(12, int)
    for i in range(12):
        if lane_count[i] > 0:
            state[i] = raw[idx] in ('G', 'g')
            idx += int(lane_count[i])
        else:
            state[i] = dummy_vals[i]
    return state

## === Main Class ======================================================
_auto_uid = 0

def _get_auto_uid():
    global _auto_uid
    uid = _auto_uid 
    _auto_uid += 1
    return uid

class SumoInteface:
    def __init__(self, fname, *, fdir=None, gui=False, cfg=None, uid=None, sil=True):
        fpath = cfg_file(fname, ".sumocfg", fdir=fdir)
        flags = ["--no-step-log", "--no-warnings"] if sil else []
        cmd   = "sumo-gui" if gui else "sumo"
        uid   = _get_auto_uid() if uid is None else uid

        traci.start([cmd, "-c", fpath, *flags], label=uid)
        self._sim = traci.getConnection(uid)

        cfg = cfg or {}
        self._init_fields(cfg.get("lane_count", None))

    def __del__(self):
        self._sim.close()

    def _init_fields(self, lane_count=None):
        # Fields to track
        self._node = NODE
        # self._lane_names  = tuple(road_name(i) for i in range(4))

        # 0) For futureproofing???
        self._lane_count  = np.zeros(12, dtype=int) # np.ones(12, dtype=int) if lane_count is None else lane_count

        # 1) Current and Next traffic light state
        self._lights      = np.zeros(12, dtype=int)
        self._lights_next = np.zeros(12, dtype=int)

        # 2) Time sensor was occupied by curr. car, name of sensor
        self._occupied_t   = np.zeros((12, 5), dtype=float)
        self._occupied_p   = np.zeros((12, 5), dtype=float)
        self._sensor_names = np.zeros((12, 5), dtype='U40')

        # 3) Count per direction (into road) inside junction or just left
        self._inside      = np.zeros(4, dtype=int)
        self._left        = np.zeros(4, dtype=int)

        # 4) Dictionary of car "state" to track if in junction or not
        self._pending  = np.zeros(12, dtype=np.uint32)
        self._cars     = {} # See _update_junction

        # 5) Number of cars added - for adding them
        self._car_index = -1

        # 6) Basic other housekeeping
        self._time = 0.0
        self._time_delta = 0.0
        self._collisions = 0

        # Retrieve all from sim
        self._init_routes()
        self._init_sensors()
        self._update_fields()

    def _init_routes(self):
        # 1) Register routes that a car can take
        for start, end in permutations(range(4), 2):
            edges = (edge_name(start, -1), edge_name(-1, end))
            name  = route_name(start, end)
            self._sim.route.add(name, edges)
        # 2) Register links on each lane
        for lane_i in range(12):
            lane = lane_name(lane_i)
            links = self._sim.lane.getLinks(lane)
            # TODO - Check directions for future reference
            self._lane_count[lane_i] = len(links)
        # for road in range(4):
        #     name = road_name(road)
        #     print(blue("For", name))
        #     print(self._sim.edge.getLinks(name))


    def _init_sensors(self):
        discovered = [[] for i in range(12)]
        for loop_id in self._sim.inductionloop.getIDList():
            # Skip traffic light sensors if necessary
            if DROP_TL and loop_id[:3] == "TLS":
                continue
            # Check which lane the sensor belongs to
            lane_name = self._sim.inductionloop.getLaneID(loop_id)
            index = lane_index(lane_name)
            if index is None:
                print(warn("_init_sensors"), "Found unusable induction loop", loop_id, lane_name)                
                continue
            # Keep a priority queue to order sensors appropriately in array
            pos = self._sim.inductionloop.getPosition(loop_id)
            heapq.heappush(discovered[index], (-pos, loop_id))
        # Iterate over sensors from closest to farthest from junction
        rows, columns = self._sensor_names.shape
        for lane in range(rows):
            for sensor in range(columns):
                if len(discovered[lane]) == 0:
                    # print(warn("_init_sensors"), "Fewer sensors than expected!", lane)
                    break
                _, loop_id = heapq.heappop(discovered[lane])
                self._sensor_names[lane, sensor] = loop_id

    def _update_fields(self):
        self._update_time()
        self._update_lights()
        self._update_sensors()
        self._update_junction()

    def _update_time(self):
        self._time = self._sim.simulation.getTime()
        self._time_delta = self._sim.simulation.getDeltaT()

    def _update_lights(self):
        raw = self._sim.trafficlight.getRedYellowGreenState(self._node)
        self._lights = unpack(raw, self._lane_count, self._lights_next)
        self._lights_next = self._lights

    def _update_sensors(self):
        # Get induction loop readings
        rows, columns = self._sensor_names.shape
        for lane in range(rows):
            for sensor in range(columns):
                loop_id = self._sensor_names[lane, sensor]
                # Skip elements where sensor doesn't exist
                if len(loop_id) == 0:
                    continue
                last    = self._occupied_t[lane, sensor] > 0
                data    = self._sim.inductionloop.getVehicleData(loop_id)
                n_cars  = self._sim.inductionloop.getLastStepVehicleNumber(loop_id)
                if len(data) == 0:
                    self._occupied_t[lane, sensor] = 0
                    self._occupied_p[lane, sensor] = 0
                else:
                    data = self._sim.inductionloop.getVehicleData(loop_id)
                    if len(data) == 0:
                        print(alarm("_update_sensors"), "Could note retrieve vehicle data")
                        exit_time = self._time
                    else:
                        car, length, enter_time, exit_time, vtype = data[-1]
                    if exit_time == -1:
                        self._occupied_t[lane, sensor] = self._time - enter_time
                        self._occupied_p[lane, sensor] = 1
                    else:
                        self._occupied_t[lane, sensor] = 0
                        portion = self._sim.inductionloop.getLastStepOccupancy(loop_id)
                        self._occupied_p[lane, sensor] = portion / 100     
    
    def _update_junction(self):
        # Collisions only occur in the junction
        self._collisions = self._sim.simulation.getCollidingVehiclesNumber()
        # _cars is dict of {car_id: [start, in_junction], ...}
        for v in self._sim.simulation.getDepartedIDList():
            start, end = route_targets(self._sim.vehicle.getRouteID(v))
            self._cars[v] = (start, False)
            lane = route_index(start, end)
            self._pending[lane] -= 1

        self._inside = np.zeros(4)
        self._left   = np.zeros(4)
        for v in self._sim.vehicle.getIDList():
            # res is None for cars already past intersection
            res = self._cars.get(v, None)
            if res is None:
                continue
            start, already_in = res
            lane_name  = self._sim.vehicle.getLaneID(v)
            # Cars inside the junction
            if lane_name[0] == ":":
                self._inside[start] += 1
                if not already_in:
                    self._cars[v] = (start, True)
            elif lane_name[:2] == NODE:
                # Already skipped from `if res is None: continue`
                # if already past intersection
                self._left[start] += 1
                self._cars.pop(v, None)

    def _get_next_state(self, lights=None):
        return self._node, pack(lights or self._lights_next, self._lane_count)

    def _next_car(self):
        self._car_index += 1
        return car_name(self._car_index)

    # def _roll_back_car(self, name):
    #     index = car_index(name)
    #     if index == self._car_index:
    #         self._car_index -= 1

    def _validate_pending(self, lane_index=None):
        if lane_index is not None:
            expected = self._pending[lane_index]
            found    = len(self._sim.lane.getPendingVehicles(lane_name(lane_index)) or [])
            if expected != found:
                notify_error(RuntimeError, "_validate_pending", expected, found)
        else:
            pending = np.zeros(12)
            for i in range(12):
                pending[i] = self._sim.lane.getPendingVehicles(lane_name(i) or [])
            if (pending != self._pending).all():
                notify_error(RuntimeError, "_validate_pending", pending, self._pending)

    # === Step ===
    def step(self, lights=None):
        node, raw = self._get_next_state(lights)
        self._sim.trafficlight.setRedYellowGreenState(node, raw)
        self._sim.simulationStep()
        self.recompute()
    
    def recompute(self):
        self._update_fields()

    # === Initial State Control ===
    def road_length(self, start):
        return self._sim.lane.getLength("{}_0".format(edge_name(start, -1)))

    def insert_car_at(self, start, end, pos):
        start = NAMES[start] if isinstance(start, int) else start
        end = NAMES[start] if isinstance(end, int) else end
        lane = route_index(start, end)
        turn = lane % 3
        route = route_name(road_index(start), road_index(end))
        if 0 < pos < 1:
            pos = pos * self.road_length(start)
        self._sim.vehicle.add(self._next_car(),
                              route,
                              departLane=turn,
                              departPos=pos)
        # To avoid issues later...
        self._pending[lane] += 1

    def insert_car_turning_at(self, start, turn, rel_pos):
        self.insert_car_at(start, turn_target(start, turn), rel_pos)

    def set_step_time(self, delta):
        self._sim.simulation.setDeltaT(d_time)
    
    def get_step_time(self):
        return self._time_delta 

    # === Flow Control ===
    def add_cars(self, routes, allow_pending=True):
        return all(self.add_car(start, end, allow_pending) for start, end in routes)

    def add_car(self, start, end, allow_pending=True):
        start = NAMES[start] if isinstance(start, int) else start
        end = NAMES[end] if isinstance(end, int) else end
        lane  = route_index(start, end)
        turn  = lane % 3
        if not allow_pending and self._pending[lane] > 0:
            # notify_error(RuntimeError, "add_car", "Already pending a car in said lane")
            return False
        route = route_name(road_index(start), road_index(end))
        self._sim.vehicle.add(self._next_car(), route, departLane=turn)
        self._pending[lane] += 1
        return True

    def add_car_turning(self, start, turn, allow_pending=True):
        return self.add_car(start, turn_target(start, turn), allow_pending)

    def get_cars_pending(self, lane_index=None):
        if lane_index is not None:
            return self._pending[lane_index]
        return self._pending

    # def set_speed_out(self, road, speed):
    #     raise NotImplementedError()
    
    # def get_speed_out(self, road):
    #     raise NotImplementedError()

    # === Action Control ===
    def set_lights(self, lights):
        if len(lights) != 12:
            notify_error(ValueError, "set_lights", "Must be 12 elements!")
        self._lights_next = np.array(lights)

    def get_lights(self):
        return self._lights

    # === Non-Controlled ===
    def get_time(self):
        return self._time

    def get_occupied(self):
        return (self._occupied_t > 0.0).astype(int)

    def get_occupied_time(self):
        return self._occupied_t

    def get_occupied_portion(self):
        return self._occupied_p

    def get_in_intersection(self):
        return self._inside

    def get_left_intersection(self):
        return self._left

    def get_collisions(self):
        return self._collisions
        # return self._sim.simulation.getCollidingVehiclesNumber()

    # === For testing ===
    def visualize(self):
        lights = self.get_lights()
        occ    = self.get_occupied()
        inside = self.get_in_intersection()
        for i in range(12):
            for o in occ[i]:
                print(colbg(" ", 'y' if o else None), end="|")
            l = lights[i]
            print(colbg("o", 'g' if l else 'r'), end="")
            if i % 3 == 1:
                print("x>" if inside[i // 3] else " >", dim(NAMES[i // 3]))
            else:
                print() # empty newline

    # === Visualization Retrieval ===
    # def shape_intersection(self):
    #     raise NotImplementedError()

    # def shape_lane(self):
    #     raise NotImplementedError()

    # def shape_car(self, road):
    #     raise NotImplementedError()

    # def shape_sensors(self):
    #     raise NotImplementedError()
    
if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="map_1", help="SUMO file to use")
    parser.add_argument("-g", "--gui", action="store_true", help="Whether to show GUI")
    parser.add_argument("-l", "--length", type=int, default=100, help="Length of episode in steps")
    args = parser.parse_args()

    sim = SumoInteface(args.file, gui=args.gui)
    for i in range(args.length):
        if i % 2 == 0:
            sim.add_car(i % 3, 3)
        sim.step()
        if i % 5:
            sim.visualize()
        if args.gui:
            time.sleep(0.1) # Easier to watch