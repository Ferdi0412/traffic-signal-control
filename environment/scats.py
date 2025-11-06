import numpy as np

from sumo_interface import SumoInterface

C_MIN    = 62
C_MAX    = 180
G_MIN    = 15

TURN_TIME = 10
CLEARANCE_TIME = 6

class SCATS:
    def __init__(self, sim):
        self.sim = sim

        self.DS_history = [0.5, 0.5] # N/S, E/W

        # Used to filer the pressure in either direction
        self.lanes = [
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5, 9, 10, 11]
        ] 

        self.roads = [
            [0, 2],
            [1, 3]
        ]

        # self.turn_lanes = [
            # [2, 8],
            # [5, 11]
        # ]

        self.lights = [
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
        ]

        self.turn_lights = [
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        ]

        self.dir = 0

    def calc_dos(self, dir):
        """Calculate Degree of Saturation.

        Return value is for entire direction, not turn/straight separate.

        NOTE: SCATS uses flow rates, not occupancy percentages, this
              is just an approximation to make use of the same data
              as provided to the network
        """
        occupied   = sim.get_occupied()
        queue_lens = sim.get_queue_length()

        occupied = np.sum(occupied[self.lanes[dir], :])
        queued   = np.sum(queue_lens[self.lanes[dir]])

        # A full lane is 25 cars, but use 15 as "full" - this is 3x the
        #                                                ratio of cars 
        #                                                queued to sensors
        return min(max(occupied, queued) / (15 * len(self.lanes[dir])), 1)

    def calc_cycle_t(self, dos1, dos2):
        max_dos = max(dos1, dos2)
        return C_MIN + (C_MAX - C_MIN) * min(max_dos, 1)

    def calc_split(self, dos1, dos2, cycle_t):
        # Total green for fwd dir.
        net_green = cycle_t - 2 * CLEARANCE_TIME - 2 * TURN_TIME

        if np.isclose(dos1 + dos2, 0):
            ns, ew = net_green / 2, net_green / 2
        else:
            ns = net_green * dos1 / (dos1 + dos2)
            ew = net_green * dos2 / (dos1 + dos2)

        ns = max(ns, G_MIN)
        ew = max(ew, G_MIN)

        # In case the max(...) changed the values
        if ns + ew > net_green:
            ns = ns * net_green / (ns + ew)
            ew = ew * net_green / (ns + ew)
        
        return ns, ew

    def half_cycle(self, cost_is_queue=True):
        """Run a half-cycle - set lights for 1 direction, then compute next"""
        old_queue = self.sim.get_queue_length()
        old_time  = self.sim.get_time()

        dos1 = self.calc_dos(0)
        dos2 = self.calc_dos(1)

        cycle_t = self.calc_cycle_t(dos1, dos2)

        green_times = self.calc_split(dos1, dos2, cycle_t)

        self.sim.set_lights(self.lights[self.dir])

        end_time = self.sim.get_time() + green_times[self.dir]

        while self.sim.get_time() < end_time:
            self.sim.step()

        self.sim.set_lights(self.turn_lights[self.dir])

        end_time = self.sim.get_time() + TURN_TIME

        while self.sim.get_time() < end_time:
            self.sim.step()

        self.dir = int(not self.dir)

        new_queue = self.sim.get_queue_length()
        new_occ   = self.sim.get_occupied()
        new_time  = self.sim.get_time()

        if cost_is_queue:
            cost = np.sum(new_queue - old_queue)
        else:
            cost = np.sum(new_occ > 5)
        elapsed = new_time - old_time

        print(cost, elapsed)
        return cost * elapsed


if __name__ == "__main__":
    sim = SumoInterface("map_1", gui=True)
    sim.set_car_prob([2 / 12] * 12)

    controller = SCATS(sim)

    for i in range(1000):
        ("For the last iter, penalty was", controller.half_cycle(True))