# Looking to ME5406_exampleSAPP:
The only thing that is used from the environment is the `SAPPEnv` class, see [ME5406_exampleSAPP.py](https://github.com/marmotlab/ME5406_exampleSAPP/blob/main/ME5406_exampleSAPP.py).

Note that the version referenced is as of commit *3ef651c*.

A single instance of `SAPPEnv` is owned by `Worker`, which is set to field `Worker.env`:
```py
# Snippet from code
# Starting from Line 32
class Worker:
    def __init__(self, game, global_network):
        self.env = game # <<- This line assigns the SAPPEnv instance
        self.global_network = global_network
        ...
```

The main methods of `Worker` are:
* `train` which runs an iteration of training the network
* `shouldRun` which simply checks when an episode should end
* `work` which executes everything

The `Worker` references the `SAPPEnv` only within the `Worker.work` method:

```py
# The work function starts from 91

# === reset() ===
# In line 102
# For every new episode, reset the env
# Get current state, and available actions
s, validActions = self.env.reset()

# === _render(...) ===
# Line 111
# For every new episode, create a buffer of rendered frames
# Called if a `Worker.work` argument/parameter is set True
episode_frames = [self.env._render(...)]

# === step(action) ===
# Line 143
# Step the environment with deisred action
# Get new state, reward (increment), simulation done (ended?), and available actions
s1, r, d, validActions = self.env.step(a)

# === _render(...) again ===
# Line 146
# After step, add another rendered frame
# Again, called only if a `Worker.work` argument is set True
episode_frames.append(self.env._render(...))
```