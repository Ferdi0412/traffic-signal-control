# Traffic Signal Control

This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors

- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

## Setup

The only new dependency from last submission was **PyTorch**. This was added to *requirements.txt*. To create the enviornment as needed, run:

```sh
conda env create -f environment.yml
```

It is our understanding that **mamba** should work as a direct replacement by simply swapping `conda` for `mamba`, though we are using conda so the actual syntax for this we have not tested.

If you are using **conda**, you can use our automated scripts:

```sh
chmod +x reset.sh
./reset.sh
```

Next, activate your conda (or mamba) environment, which will be named **traffic**:

```sh
conda activate traffic
```

## Running the Code

Run the NN code:

```sh
python project/neural_network.py
```

## Note

Due to SUMO (the actual simulator, not our code) implementation for **traci** which I believe uses C++ threads, you will likely have at the end of the code a message similar to the following:

"<span style="color: red">**SumoInterface.\_\_del\_\_**</span> error('required argument is not an integer')"

"Error: tcpip::Socket::recvAndCheck @ recv: ..."

This is **not an issue**, just a quirk of the library we use. It will in no way impact the training, as this ONLY EVER occurs when python is cleaning up the modules.

## File Structure

```txt
/
├── project/                   # This is where we keep all our main code at the moment
│   │                          # "project/" was previously "environment/
│   ├── import_sumo.py         # traci import issue workaround
│   ├── sumo_interface.py      # SumoInterface definition
│   ├── traffilightgymsumo.py  # Gym code
│   ├── neural_network.py       # NN implementation 
├── sumo_networks/             # This is where the SUMO config files are
```