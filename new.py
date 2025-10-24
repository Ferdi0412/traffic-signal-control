import os, sys
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

path = os.path.join(os.path.dirname(__file__), "sumo-networks", "map_1.sumocfg")

print(path)
traci.start(["sumo-gui", "-c", path])

while True:
    traci.stepSimulation()