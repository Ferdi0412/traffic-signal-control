import numpy as np

from sumo_interface import SumoInterface
from giffer import SumoGif
import os
from datetime import datetime

C_MIN    = 62
C_MAX    = 180
G_MIN    = 15

TURN_TIME = 10
CLEARANCE_TIME = 6

class SCATS:
    def __init__(self, sim):
        self.sim = sim
        self.save_dir = "./"
        self.DS_history = [0.5, 0.5] # N/S, E/W
        
        # Variables initialization for comparison with DQN
        self.new_queue = np.zeros(12, dtype=int)
        self.prev_queue_length = np.zeros(12, dtype=int)
        self.reward_weights = [0.01, 0.03]
        self.compare_reward = 0.
        self.compare_deltaq = 0.
        self.compare_longwait = 0.
        self.counter = 0
        self.total_qlength = np.zeros(12, dtype=int)
        self.average_qlength = np.zeros(12, dtype=int)
        self.total_waittime = np.zeros(12, dtype=float)
        self.throughput = np.zeros(4, dtype=float)

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
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
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
        occupied   = self.sim.get_occupied()
        queue_lens = self.sim.get_queue_length()

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

        self.prev_queue_length = self.sim.get_queue_length()

        while self.sim.get_time() < end_time:
            self.sim.set_car_prob([1 / 12] * 12)
            self.sim.step()
            self.counter += 1
            qlength = self.sim.get_queue_length()
            self.total_qlength += qlength
            ep_waittime = self.sim.get_occupied_time()
            waittime = np.sum(ep_waittime, axis=1)
            self.total_waittime += waittime
            outgoing = self.sim.get_left_intersection()
            self.throughput += outgoing
            if self.render:
                self.sim._update_gif()
            # To prevent overshooting of simulation time
            if self.sim.get_time() - 5 >= self.max_simtime:
                break

        self.sim.set_lights(self.turn_lights[self.dir])

        end_time = self.sim.get_time() + TURN_TIME

        while self.sim.get_time() < end_time:
            self.sim.set_car_prob([1 / 12] * 12)
            self.sim.step()
            self.counter += 1
            qlength = self.sim.get_queue_length()
            self.total_qlength += qlength
            ep_waittime = self.sim.get_occupied_time()
            waittime = np.sum(ep_waittime, axis=1)
            self.total_waittime += waittime
            outgoing = self.sim.get_left_intersection()
            self.throughput += outgoing
            if self.render:
                self.sim._update_gif()
            # To prevent overshooting of simulation time
            if self.sim.get_time() - 5 >= self.max_simtime:
                break

        self.dir = int(not self.dir)

        self.new_queue = self.sim.get_queue_length()
        new_occ   = self.sim.get_occupied()
        new_time  = self.sim.get_time()

        total_reward, delta_qlength,penalty_longwait = self.generate_rewards(self.reward_weights)

        if cost_is_queue:
            cost = np.sum(self.new_queue - old_queue)
        else:
            cost = np.sum(new_occ > 5)
        elapsed = new_time - old_time

        #print(cost, elapsed)
        run_time = self.sim.get_time()
        return run_time, total_reward, delta_qlength, penalty_longwait

    def generate_rewards(self,reward_weights):
        """Generate results based on what was used in DQN training for purpose of comparison"""
        w1 = reward_weights[0]
        w2 = reward_weights[1]
        # penalty_wait = 0
        penalty_longwait = 0
        
        delta_qlength = int(self.prev_queue_length.sum() - self.new_queue.sum())

        cars_waitinglong = np.sum(self.sim.get_occupied_time()>60)
        
        delta_qlength = w1*delta_qlength
        penalty_longwait = -w2* cars_waitinglong

        total = delta_qlength + penalty_longwait

        return total, delta_qlength,penalty_longwait

    def single_epoch_run(self, max_simtime, render, cost_is_queue=True):
        """Run a single episode based on input simulation time and generate GIF and metrics for comparison"""
        self.max_simtime = max_simtime
        self.render = render
        self.run_time = 0
        
        if self.render:
            date = datetime.now().strftime("%d%m%Y_%H%M%S")
            gif_filename = os.path.join(self.save_dir, f"SCATS_{date}.gif")
            self.sim.reset(gif=gif_filename)

        while self.run_time < max_simtime:
            self.run_time, reward, delta_qlength, penalty_longwait = self.half_cycle(True)
            self.compare_reward += reward
            self.compare_deltaq += delta_qlength
            self.compare_longwait += penalty_longwait
            # To prevent overshooting of simulation time
            if self.sim.get_time() - 5 >= self.max_simtime:
                break

        self.average_qlength = self.total_qlength/self.counter

        # Save GIF if one was created
        if self.render:
            self.sim.save_gif()
            print(f"GIF saved: {gif_filename}")

    def _get_compare_rewards(self):
        """gets rewards for comparison"""
        return self.compare_reward, self.compare_deltaq, self.compare_longwait

    def _get_comparison_metrics(self):
        """gets metrics for comparison"""
        return self.average_qlength, self.total_waittime, self.throughput/self.max_simtime

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="map_2", help="SUMO file to use")
    parser.add_argument("-g", "--gui", action="store_true", help="Whether to show GUI")
    args = parser.parse_args()

    unified_config = {
        # SUMO Config
        'sumo': {
            "fname": args.file,
            "gui": args.gui,
            "seed": 8,
        },
    }

    sumo_config = unified_config['sumo']

    sumo = SumoInterface(**sumo_config)

    controller = SCATS(sumo)

    controller.single_epoch_run(1800, True, True)