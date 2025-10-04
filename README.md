# Traffic Signal Control
This project aims to train an agent to control the flow of traffic through an intersection, using reinforcement learning.

## Authors
- Goh Chian Kai
- Lam Kai Yi
- Ferdinand Tonby-Strandborg

```txt
traffic-rl-sumo/
│
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
│   ├── sumo_interface.py
│   ├── gym_wrapper.py
│   ├── reward_calculator.py
│   └── state_processor.py
│
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py
│   ├── baseline_agents.py
│   └── hyperparameters.yaml
│
├── utils/
│   ├── __init__.py
│   ├── metrics_logger.py
│   ├── visualization.py
│   └── config_loader.py
│
├── mocks/
│   ├── __init__.py
│   ├── mock_sumo_interface.py
│   └── mock_environment.py
│
├── tests/
│   ├── __init__.py
│   ├── test_sumo_interface.py
│   ├── test_reward.py
│   ├── test_gym_wrapper.py
│   └── test_integration.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── demo.py
│   ├── generate_routes.py
│   └── visualize_results.py
│
├── configs/
│   ├── experiment_config.yaml
│   └── paths.yaml
│
├── results/
│   ├── training_runs/
│   │   ├── run_001_dqn_baseline/
│   │   │   ├── config_used.yaml
│   │   │   ├── model_checkpoints/
│   │   │   │   ├── model_step_10000.zip
│   │   │   │   ├── model_step_20000.zip
│   │   │   │   └── model_final.zip
│   │   │   ├── logs/
│   │   │   │   ├── training_log.csv
│   │   │   │   └── metrics_log.json
│   │   │   └── tensorboard/
│   │   │       └── events.out.tfevents
│   │   │
│   │   └── run_002_dqn_tuned/
│   │       ├── config_used.yaml
│   │       ├── model_checkpoints/
│   │       ├── logs/
│   │       └── tensorboard/
│   │
│   ├── evaluation/
│   │   ├── metrics_summary.csv
│   │   └── comparison_plots/
│   │       ├── rewards_comparison.png
│   │       └── queue_length_comparison.png
│   │
│   └── videos/
│       ├── demo_episode_001.mp4
│       └── demo_episode_002.mp4
│
├── docs/
│   ├── setup_guide.md
│   ├── sumo_network_design.md
│   ├── rl_algorithm_design.md
│   └── architecture_diagram.png
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
```