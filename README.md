# Traffic Signal Control
This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors
- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

# Setup

## Cloning
This is the process of "downloading" the GitHub repository (remote repository), while retaining the ability to **push** and **pull** changes to/from it.

### SSH Key Generation
Ensure that you have the appropriate SSH keys for using GitHub. Ensure that you remember in which directory the files are created (typically `C:/Users/xxx/.ssh/` for Windows or `~/.ssh/` for Linux).
```sh
ssh-keygen -t ed25519
```

Then copy the contents of the `id_ed25519.pub` file, and go to your GitHub accounts `settings > SSH and GPG keys` (*SSH and GPG keys* are under the *Access* grouping in the settings page), and press `New SSH key`. Give it an appropriate *Title* and paste the contents of the `id_ed25519.pub` file there (leave the *Key type* as *Authentication Key*). For linux, you can quickly see the contents of this file in terminal using:
```sh
cat ~/.ssh/id_ed25519.pub # Copy the entire output
```

### Git Config
In order to clone a GitHub repository, there are 2 key config items. The first is your **name**. GitHub will accept p. much anything, but use something recognizable (like your full name). The second is you **email**. This must be tied to your GitHub account, otherwise cloning repositories will not work.
```sh
git config --global user.name "Full Name"
git config --global user.email "github.account.email@domain.com
```

### Git Clone
Now you can clone the GitHub repository. The *target* of this command is standardized for GitHub repositories as `git@github.com:USER/REPOSITORY.git`, where `USER` is the username for the account hosing the repository. For this repository, the `USER` is `Ferdi0412`, and the `REPOSITORY` is `traffic-signal-control`.
```sh
git clone git@github.com:Ferdi0412/traffic-signal-control
```

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

## File Structure

```txt
traffic-rl-project/
├── requirements.txt
├── environment.yml
├── README.md
├── .gitignore
│
├── contracts/
│   └── interfaces.py          # Shared definitions (ALL edit this)
├── environment/
│   ├── sumo_interface.py      # Ferdi owns
│   ├── gym_wrapper.py         # KY C owns
│   └── reward.py              # CK B owns
├── agents/
│   └── dqn_agent.py           # CK owns
├── sumo_network/              # Ferdi owns (xml and sumocfg files)
├── tests/                     # Everyone writes tests (optional)
│   ├── test_sumo.py           # Ferdi owns 
│   ├── test_reward.py         # CK owns 
│   └── test_integration.py    # KY owns 
└── demo.py                    # KY owns
```