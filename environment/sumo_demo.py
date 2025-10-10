import os
import sys
import time
import argparse
from itertools import permutations

import numpy as np

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

NODE_ID   = "A0"

class BasicIntersection:
    """The important methods are:
    * [v] step
    * [ ] occupancy
    * [v] light_state
    * [v] lights
    * [v] change_light
    * ~~[intersection_state]~~
    * ~~[upstream/downstream state]~~
    * ~~[collision]~~
    """
    @staticmethod
    def node_id(name=None):
        """Get the id of a node from it's colloquial name.
        name <str> The nautical direction (eg. "N" for "North") or intersection node if None. 
        """
        if name is None:
            return NODE_ID
        if name == "N":
            return "top0"
        if name == "E":
            return "right0"
        if name == "S":
            return "bottom0"
        if name == "W":
            return "left0"
        raise ValueError

    @staticmethod
    def light_from_bool(vals):
        """Translates the lights from 0/1 to RedYellowGreenState string."""
        def _for_dir(vals):
            """For handling a direction, with 3 boolean values to RSSSL."""
            r, s, l = vals
            return ("GG" if r else "rr") + ("G" if s else "r") + ("Gg" if l else "rr")
        return _for_dir(vals[:3]) \
             + _for_dir(vals[3:6]) \
             + _for_dir(vals[6:9]) \
             + _for_dir(vals[9:12])

    @staticmethod
    def light_to_bool(string):
        """Translates the lights from RedYellowGreenState string to 0/1."""
        def _for_dir(string):
            """Get 3 boolean values from RSSSL."""
            string = string.lower()
            return [string[0] == "G", string[2] == "G", string[4] == 'g']
        return _for_dir(string[:5]) \
             + _for_dir(string[5:10]) \
             + _for_dir(string[10:15]) \
             + _for_dir(string[15:20])

    def _reg_route(self, start, end):
        """Routes are registered, for example as NE for North->East."""
        route_id = start+end
        edges    = (self.node_id(start)+NODE_ID, NODE_ID+self.node_id(end))
        self._sim.route.add(route_id, edges)

    def _read_ind_loops(self):
        dirs = ('N', 'E', 'S', 'W')
        for i, direction in enumerate(dirs):
            for lane, loop_id in enumerate(self._ind_loop_ids[direction]):
                # Vehicle Data is:
                # [vehicle id, length, entry time, exit time, type]
                # print("Shape:", self._ind_loop_times.shape, "i", i, "lane", lane)
                occ_time = 0.
                data = self._sim.inductionloop.getVehicleData(loop_id)
                if len(data) > 0:
                    occ_time = self._sim_time - data[0][2]
                self._ind_loop_times[i, lane] = occ_time

    # def _reg_induction_loop(self, loop_id, lane_id):
    #     """Loops are retrieved using TraCI.
    #     Each traffic light should have an inductionloop, at least if the traffic light logic is actuated.
    #     """
    #     print("Induction loop", loop_id, "on lane", lane_id)

    def _see_lights(self):
        # List all traffic lights and their directions
        for tid in self._sim.trafficlight.getIDList():
            print("Traffic Light " + tid)
            for i, l in enumerate(self._sim.trafficlight.getControlledLinks(tid)):
                start, end, via = l[0]
                print("Light", i, "controls", start, "to", end)

    def __init__(self, file, *, uid=None, gui=False):
        """Open a SUMO simulation, using a .sumocfg file.

         Args
        file <str>  .sumocfg file
        uid  <str>  Unique simulation identifier, for parallel sims
        gui  <bool> If True, starts `sumo-gui`, else `sumo`
        """
        traci.start(["sumo-gui" if gui else "sumo", "-c", file], label=uid or "sim1")
        self._sim = traci.getConnection(uid or "sim1")

        # Backwards-lookup for colloquial names from node name
        self._lookup = {self.node_id(d): d for d in ("N", "E", "S", "W")}

        # Register all possible routes cars might take - no u-turns
        for start, end in permutations(("N", "E", "W", "S"), 2):
            self._reg_route(start, end)
        
        # Discover all induction loops
        ind_loop_pos = {
            "N": [None] * 3,
            "W": [None] * 3,
            'E': [None] * 3,
            'S': [None] * 3
        }
        self._ind_loop_ids = {
            # index by: dir -> lane
            # eg. "dir": [id_right, id_straight, id_left]
            "N": [None] * 3,
            "W": [None] * 3,
            "E": [None] * 3,
            "S": [None] * 3,
        }
        for loop_id in self._sim.inductionloop.getIDList():
            lane_id = self._sim.inductionloop.getLaneID(loop_id)
            pos = self._sim.inductionloop.getPosition(loop_id)
            # lane_len = self._sim.lane.getLength(lane_id)
            # print("Induction loop", loop_id, "is at pos", pos, "of", lane_len, "along", lane_id)
            direction = self._lookup[lane_id.split(NODE_ID)[0]]
            lane = int(lane_id.split("_")[-1])
            # For now only keep the closest sensor
            if ind_loop_pos[direction][lane] is None or pos > ind_loop_pos[direction][lane]:
                ind_loop_pos[direction][lane] = pos
                self._ind_loop_ids[direction][lane] = loop_id
        
        # Check have all desired induction loops
        for d in self._ind_loop_ids.values():
            for l in d:
                assert l is not None

        self._ind_loop_times = np.zeros((4, 3))
        self._sim_time = self._sim.simulation.getTime()

        # Get traffic light phases
        # Order is N, E, S, W
        # Current "lights" order is top, right, bottom, left
        self._curr_lights = [1] * 12
        self._new_lights  = [0] * 12

        # All cars need a unique id
        self._car_count = 0

        

    ### ===========================
    ### === IMPORTANT INTERFACE ===
    ### ===========================
    def step(self):
        # Set new states to sim
        if self._curr_lights != self._new_lights:
            self._sim.trafficlight.setRedYellowGreenState(NODE_ID, self.light_from_bool(self._new_lights))

        # Step simulation
        self._sim.simulationStep()
        self._sim_time = self._sim.simulation.getTime()

        # Get new states from sim
        self._curr_lights = self.light_to_bool(self._sim.trafficlight.getRedYellowGreenState(NODE_ID))
        self._read_ind_loops()

    @property
    def occupancy_times(self):
        """Retrieves an array of the occupancy times of each induction loop."""
        return self._ind_loop_times

    @property
    def lights(self):
        """Retrieve an array of the light values, in order:
            N, E, S, W
            right, straight, left (right-hand drive)
        """
        return self._curr_lights.copy()

    @lights.setter
    def lights(self, val):
        """Set an array of the light values, in order:
            N, E, S, W
            right, straight, left (right-hand-drive)
        """
        assert isinstance(val, list) and len(val) == 12
        self._new_lights = val.copy()

    def get_light(self, dir, lane):
        """dir is one of 'N', 'S', 'E', 'W'
        lane is one of 'l', 's', 'r'
        """
        return self._curr_lights[dir * 3 + lane]

    def set_light(self, dir, lane, green):
        """Same as get_light, but green is <bool>."""
        self._new_lights[dir * 3 + lane] = 1 if green else 0

    def add_car(self, start, end):
        """start and end are 2 directions, ie. 'N', 'E', 'W', 'S'."""
        vid = "car{}".format(self._car_count)
        self._car_count += 1
        return self._sim.vehicle.add(vid, start + end)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--view", action="store_true", help="Use this flag to load the SUMO GUI.")
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = BasicIntersection(cur_dir + "/../sumo-networks/demo.sumocfg", gui=args.view)

    state_1 = [1] * 3 + [0] * 3 + [1] * 3 + [0] * 3
    state_2 = [int(not v) for v in state_1]

    sim.lights = state_1

    routes = list(permutations(('N', 'E', 'W', 'S'), 2))

    for i in range(100):
        if i % 10 == 0:
            print("Iteration", i)
            print("Current induction loop/sensor times:")
            print(sim.occupancy_times)
            # print("Last state:", sim.lights)
            if i % 20 == 0:
                sim.lights = state_1
            else:
                sim.lights = state_2

        # Add a car in 2 directions
        for i in range(np.random.randint(1, 3)):
            start, end = routes[np.random.randint(0, len(routes))]
            # print("Adding car from", start, "to", end)
            sim.add_car(start, end)

        sim.step()
        time.sleep(0.2) # 200ms delay