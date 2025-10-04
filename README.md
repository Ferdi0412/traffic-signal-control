# Traffic Signal Control
This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors
- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

  traffic-rl-sumo/
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── contracts/
│   ├── interfaces.py
│   └── CONTRACTS.md
│
├── sumo_files/
│   ├── networks/
│   │   ├── four_way_junction.net.xml
│   │   ├── four_way_junction.nod.xml
│   │   ├── four_way_junction.edg.xml
│   │   ├── four_way_junction.con.xml
│   │   └── four_way_junction.tll.xml
│   │
│   ├── routes/
│   │   ├── low_demand.rou.xml
│   │   ├── medium_demand.rou.xml
│   │   ├── high_demand.rou.xml
│   │   └── rush_hour.rou.xml
│   │
│   ├── configs/
│   │   ├── training.sumocfg
│   │   ├── testing.sumocfg
│   │   └── demo.sumocfg
│   │
│   └── additional/
│       ├── vehicle_types.add.xml
│       └── detector_config.add.xml
│
├── environment/
│   ├── __init__.py
│   ├── sumo_interface.py        # Person A owns
│   ├── gym_wrapper.py            # Person C owns
│   ├── reward_calculator.py      # Person B owns
│   └── state_processor.py        # Person A owns
│
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py              # Person B owns
│   ├── baseline_agents.py        # Person B owns (random, fixed-timing)
│   └── hyperparameters.yaml      # Person B owns
│
├── utils/
│   ├── __init__.py
│   ├── metrics_logger.py         # Person C owns
│   ├── visualization.py          # Person C owns
│   └── config_loader.py          # Person C owns
│
├── mocks/
│   ├── __init__.py
│   ├── mock_sumo_interface.py    # Person C creates for testing
│   └── mock_environment.py       # Person B creates for testing
│
├── tests/
│   ├── __init__.py
│   ├── test_sumo_interface.py    # Person A writes
│   ├── test_reward.py            # Person B writes
│   ├── test_gym_wrapper.py       # Person C writes
│   └── test_integration.py       # All collaborate
│
├── scripts/
│   ├── train.py                  # Person B owns
│   ├── evaluate.py               # Person C owns
│   ├── demo.py                   # Person C owns
│   ├── generate_routes.py        # Person A owns
│   └── visualize_results.py      # Person C owns
│
├── configs/
│   ├── experiment_config.yaml    # Shared configuration
│   └── paths.yaml                # File paths configuration
│
├── results/
│   ├── training_runs/
│   │   ├── run_001_dqn_baseline/
│   │   │   ├── model_checkpoints/
│   │   │   ├── logs/
│   │   │   └── tensorboard/
│   │   └── run_002_dqn_tuned/
│   │
│   ├── evaluation/
│   │   ├── metrics_summary.csv
│   │   └── comparison_plots/
│   │
│   └── videos/
│       └── demo_episode_001.mp4
│
├── docs/
│   ├── setup_guide.md
│   ├── sumo_network_design.md    # Person A writes
│   ├── rl_algorithm_design.md    # Person B writes
│   └── architecture_diagram.png  # Person C creates
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
│
└── report/
    ├── final_report.pdf
    └── sections/
        ├── 01_introduction.md
        ├── 02_environment_design.md
        ├── 03_rl_algorithm.md
        ├── 04_results.md
        └── 05_reflections.md
