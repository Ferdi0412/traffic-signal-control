# GROUP 39: Traffic Signal Control

This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

> The report is in this repository, named **Group 39 Learning Agent Report.pdf**

## Authors

- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

# Setup
## Setting up SUMO and Python
Our environment is named `traffic`:
```sh
conda activate traffic
```

> No new dependencies since submission 2.
> If you don't have the previous environment for this project (`traffic`): `chmod +x reset.sh && ./reset.sh`


## Running the Code
Run the learning agent code:

```sh
python project/LearningAgent.py
```

### Note
Due to the SUMO library we use, `TraCI`, and the way `Pillow` stores graphics (from previous submissions), you may an **error message** similar to:

"<span style="color: red">**SumoInterface.\_\_del\_\_**</span> error('required argument is not an integer')"

"Error: tcpip::Socket::recvAndCheck @ recv: ..."

> This only happens randomly AFTER our code is done, due to the way these dependencies are implemented with threading, and is beyond our control - the only fix is to `time.sleep` and hope they end in the correct order

### Optional
> This is not meant for our submission!

If you have a Weights And Biases account, you can run the code with the following flag, though this requires further setup:

```sh
# Not meant for submission 3
pip install wandb

wandb login

python project/LearningAgent.py --wandb
```

## File Structure

```txt
/
├── "Group 39 Learning Agent Report.pdf"
├── project/                   # This is where we keep all our main code at the moment
│   │
│   ├── import_sumo.py         # traci import issue workaround
│   ├── sumo_interface.py      # SumoInterface definition
│   ├── traffilightgymsumo.py  # Gym code
│   ├── LearningAgent.py       # NN and RL Agent implementation 
├── sumo_networks/             # This is where the SUMO config files are, as needed by the sim
```