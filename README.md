# Traffic Signal Control

This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors

- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

# Setup

## Setting up SUMO and Python

After cloning, run the setup.sh script

Shell scripts need permissions to run

```sh
chmod +x setup.sh
```

Run the script

```sh
./setup.sh
```

Activate your conda environment

```sh
conda activate traffic
```

## Running the Code

Ensure that you are in the traffic-signal-control folder

```sh
cd traffic-signal-control/
```

### With GUI

Run the demo code

```sh
python environment/trafficlightgymsumo.py --gui
```

The SUMO gui should appear and you should see a cross intersection in front of you
![SUMO Gui](image.png)

Change the delay at the top to 100ms and click the play button. The simulation should run.

When it ends, proceed to click yes to closing all open files and views.

You can then view the outputs of the simulation every 10 steps in your terminal.

### Without GUI

If you wish to run the demo code without the gui, type this in the terminal instead.

```sh
python environment/trafficlightgymsumo.py
```

You can now view the outputs of the simulation every 10 steps in your terminal without the GUI popping up.

## File Structure

```txt
traffic-rl-project/
├── requirements.txt
├── environment.yml
├── README.md
├── .gitignore
├── environment/
│   ├── sumo_interface.py
│   ├── traffilightgymsumo.py
│   ├── utils.py
├── sumo_network/
```
