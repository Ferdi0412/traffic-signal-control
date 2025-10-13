import numpy as np

"""
STATE SPACE

"traffic_light_state": array of 4x3x3 of true or false
    - 4 intersections, each with 3 traffic lights (left, middle, right)
    - each traffic light has 3 possible directions (turn left, go straight, turn right)
    - true = green, false = red
    - state["traffic_light_state"][dirindex][laneindex][turnindex]

"occupancy_lane": array of 4x3x12 of bool
    - 4 intersections, each with 3 lanes, each lane with 12 segments
    - true = occupied, false = unoccupied
    - state["occupancy_lane"][dirindex][laneindex][segmentindex]
    - segmentindex = 0 is closest to intersection, segmentindex = 11 is farthest

"occupancy_intersection" : 6x6 array of bool
    - 6x6 grid representing the intersection area
    - true = occupied, false = unoccupied
    - state["occupancy_intersection"][x][y]

"waiting_time" : array of 4x3x12 of int
    - 4 intersections, each with 3 lanes, each lane with 12 segments
    - integer representing the waiting time of the vehicle in that segment
    - state["waiting_time"][dirindex][laneindex][segmentindex]
    - segmentindex = 0 is closest to intersection, segmentindex = 11 is far

"downstream_status" : array of 4
    - 4 intersections
    - integer representing the number of vehicles in the downstream intersection
    - state["downstream_status"][dirindex]

"upstream_status" : array of 4
    -  4 intersections
    - integer representing the number of vehicles in the upstream
    - state["upstream_status"][dirindex]

ACTION SPACE

"traffic_light_status" : array of 4x3x3 of true or false
    - 4 intersections, each with 3 traffic lights (left, middle, right)
    - each traffic light has 3 possible directions (turn left, go straight, turn right)
    - true = green, false = red
    - action["traffic_light_status"][dirindex][laneindex][turnindex]

"""

# Direction index: dirindex
N = 0
E = 1
S = 2
W = 3

#Lane index: laneindex
left = 0
middle = 1
right = 2

# Turn index: turnindex
turn_left = 0
go_straight = 1
turn_right = 2

state = {
    "traffic_light_state": np.zeros((4,3,3), dtype = bool),# array of 4x3 of true or false
    "occupancy_intersection" : np.zeros((6,6),dtype=bool),#6x6 array of 0,1 (can change)
    "waiting_time" : np.zeros((4,3,12), dtype = int),
    "downstream_status" : np.zeros(4, dtype = int), #array of 4 of int
    "upstream_status" : np.zeros(4, dtype = int) #array of 4 of int
}

action = {
    "traffic_light_status" : np.zeros((4,3,3), dtype = bool) # array of 4x3 of true or false
}


