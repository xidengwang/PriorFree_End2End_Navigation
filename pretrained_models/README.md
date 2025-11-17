# Pre-trained Models

This directory contains all the pre-trained models necessary to reproduce the simulation results presented in our paper. Using these models allows for direct evaluation without the need for time-consuming retraining.

The models are organized into two main categories based on their usage.

---

## 1. Reinforcement Learning Agent Models

**Path:** `pretrained_models/rl_agent/`

These models are the core components of our proposed USV motion planner and are used for all primary navigation tasks and comparisons against other planners (like DWA and NRMP).

### 📄 `sac_ship_planner_final.zip`

*   **Description:** The final trained SAC (Soft Actor-Critic) agent. This file contains the learned policy and value networks that generate navigation commands.
*   **Framework:** Stable Baselines3.
*   **Usage:** This model is loaded by all main evaluation scripts, including:
    *   `experiments/02_comparison_with_dwa/run_dwa_comparison.py`
    *   `experiments/04_non_convex_environment/run_non_convex_test.py`

### 📄 `perception_net_ship.pth`

*   **Description:** The Perception Network trained specifically for the **USV's hull shape**. It provides fast distance estimations from sensor points to the USV's boundary, which is a critical input for the RL environment.
*   **Model Type:** PyTorch `nn.Module`.
*   **Usage:** This network is loaded automatically by the `MyShipEnv` environment (`ship_rl_planner/environment.py`) during initialization. It is used in all experiments involving our planner.

---

## 2. Comparison Experiment Models (Acker Vehicle)

**Path:** `pretrained_models/comparison_models/acker_vehicle/`

These models are used **exclusively** for the perception network performance comparison experiment (Figure X in the paper). This benchmark is based on a standard Ackermann steering vehicle model, which has a different geometric shape than our USV.

### 📄 `perception_net_acker.pth`

*   **Description:** The Perception Network trained for the **Ackermann vehicle's rectangular shape**. This model is used to demonstrate the accuracy and speed of our perception method in the comparison benchmark.
*   **Model Type:** PyTorch `nn.Module`.
*   **Usage:** Loaded by the `experiments/01_perception_net_vs_dune/run_perception_comparison.py` script as the "Perception Net" method.

### 📄 `dune_acker.pth`

*   **Description:** The DUNE model, trained for the same Ackermann vehicle shape. This serves as the primary baseline against which our perception network is compared.
*   **Model Type:** PyTorch `nn.Module`.
*   **Usage:** Loaded by the `experiments/01_perception_net_vs_dune/run_perception_comparison.py` script as the "DUNE" method.