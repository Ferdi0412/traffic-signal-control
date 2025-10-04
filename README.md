# Traffic Signal Control
This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors
- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

# traffic-rl-sumo/

## Root Files
- README.md
- environment.yml
- requirements.txt
- .gitignore

## contracts/
Single source of truth for interfaces between modules
- interfaces.py
- CONTRACTS.md

## sumo_files/
All SUMO-specific configuration and network files (Person A owns)

### networks/
Junction topology definitions
- four_way_junction.net.xml
- four_way_junction.nod.xml
- four_way_junction.edg.xml
- four_way_junction.con.xml
- four_way_junction.tll.xml

### routes/
Traffic demand patterns
- low_demand.rou.xml
- medium_demand.rou.xml
- high_demand.rou.xml
- rush_hour.rou.xml

### configs/
SUMO simulation settings
- training.sumocfg
- testing.sumocfg
- demo.sumocfg

### additional/
Extra SUMO features
- vehicle_types.add.xml
- detector_config.add.xml

## environment/
Environment logic that RL agent interacts with

- `__init__.py`
- `sumo_interface.py` *(Person A owns)* - Direct SUMO/TraCI communication
- `gym_wrapper.py` *(Person C owns)* - OpenAI Gym API implementation
- `reward_calculator.py` *(Person B owns)* - Reward function logic
- `state_processor.py` *(Person A owns)* - Convert raw SUMO data to observations

## agents/
RL algorithm implementations (Person B owns)

- `__init__.py`
- `dqn_agent.py` - Main DQN implementation
- `baseline_agents.py` - Random, fixed-timing policies for comparison
- `hyperparameters.yaml` - All tunable parameters

## utils/
Cross-cutting utilities (Person C owns)

- `__init__.py`
- `metrics_logger.py` - Log episode data to CSV/JSON
- `visualization.py` - Plot training curves, heatmaps
- `config_loader.py` - Load YAML configs safely

## mocks/
Enable parallel development without dependencies

- `__init__.py`
- `mock_sumo_interface.py` *(Person C creates)* - Fake SUMO for testing
- `mock_environment.py` *(Person B creates)* - Fake Gym env for agent testing

## tests/
Unit and integration tests

- `__init__.py`
- `test_sumo_interface.py` *(Person A writes)* - Test TraCI commands
- `test_reward.py` *(Person B writes)* - Test reward calculations
- `test_gym_wrapper.py` *(Person C writes)* - Test Gym API compliance
- `test_integration.py` *(All collaborate)* - Test full pipeline

## scripts/
Entry points for different workflows

- `train.py` *(Person B owns)* - Start training run
- `evaluate.py` *(Person C owns)* - Evaluate trained model
- `demo.py` *(Person C owns)* - Run visual demonstration
- `generate_routes.py` *(Person A owns)* - Create traffic demand files
- `visualize_results.py` *(Person C owns)* - Generate plots from logs

## configs/
Centralized configuration files (Shared)

- `experiment_config.yaml` - High-level experiment settings
- `paths.yaml` - File path constants

## results/
Store all outputs from experiments (Generated during runs)

### training_runs/

#### run_001_dqn_baseline/
- `config_used.yaml`
- **model_checkpoints/**
  - `model_step_10000.zip`
  - `model_step_20000.zip`
  - `model_final.zip`
- **logs/**
  - `training_log.csv`
  - `metrics_log.json`
- **tensorboard/**
  - `events.out.tfevents.*`

#### run_002_dqn_tuned/
- `config_used.yaml`
- **model_checkpoints/**
- **logs/**
- **tensorboard/**

### evaluation/
- `metrics_summary.csv`
- **comparison_plots/**
  - `rewards_comparison.png`
  - `queue_length_comparison.png`

### videos/
- `demo_episode_001.mp4`
- `demo_episode_002.mp4`

## docs/
Design documentation separate from code

- `setup_guide.md` *(Person C writes)* - Installation instructions
- `sumo_network_design.md` *(Person A writes)* - Junction design rationale
- `rl_algorithm_design.md` *(Person B writes)* - Algorithm choices
- `architecture_diagram.png` *(Person C creates)* - System overview visual

## notebooks/
Jupyter notebooks for analysis (Optional)

- `exploratory_analysis.ipynb`
- `results_visualization.ipynb`

## report/
Final report assembly (Week 6, all collaborate)

- `final_report.pdf`
- **sections/**
  - `01_introduction.md`
  - `02_environment_design.md` *(Person A primary author)*
  - `03_rl_algorithm.md` *(Person B primary author)*
  - `04_results.md` *(Person C primary author)*
  - `05_reflections.md` *(All collaborate)*