import subprocess
import os

NAME = "map_1"
LENGTH = 200

# 1. Generate a grid network with traffic lights
subprocess.run([
    'netgenerate',
    '--grid',
    '--grid.number=1',
    f'--grid.attach-length={LENGTH}',
    # '--grid.length=200',
    f'--output-file={NAME}.net.xml',
    '--default.lanenumber=3',
    '--tls.guess=true',  # Automatically add traffic lights at intersections
    '--tls.default-type=static',  # Use actuated traffic lights (optional, default is static)
    '--no-turnarounds=true',  # Disable u-turns
    # '--turn-lanes=1'  # Dedicate leftmost lane for left turns only
])

# 2. Insert Induction Loops
N_SENSORS = 5

def lane_name(road, lane):
    if road > 3:
        road %= 4
    ROADS = ("top0", "right0", "bottom0", "left0")
    return "{}A0_{}".format(ROADS[road], lane)

def induction_loop(road, lane, pos):
    index = (road * 4 + lane) * N_SENSORS + pos
    name  = lane_name(road, lane)
    pos   = (LENGTH - 12.6) - (3.8 + pos * 7.5)
    template = '    <inductionLoop id="loop{}" lane="{}" pos="{:.2f}" freq="60" file="NUL"/>\n'
    return template.format(index, name, pos)

with open(f'{NAME}.net.xml', 'r') as file:
    lines = file.readlines()

lines.insert(2, "<!-- Created with netgenerate, later edited to add induction loops. -->")

line = -1
for i, l in enumerate(lines):
    if "</net>" in l:
        line = i - 1
        break

lines.insert(line, "\n")
line += 1

for road in range(4):
    for lane in range(3):
        for pos in range(N_SENSORS):
            lines.insert(line, induction_loop(road, lane, pos))
            line += 1

with open(f'{NAME}.net.xml', 'w') as file:
    file.writelines(lines)


# 3. Create config file
config = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{NAME}.net.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="360000"/>
    </time>
</configuration>"""

with open(f'{NAME}.sumocfg', 'w') as f:
    f.write(config)

print(f"Complete simulation ready! Run with: sumo-gui -c {NAME}.sumocfg")