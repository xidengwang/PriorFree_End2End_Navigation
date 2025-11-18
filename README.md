# An End-to-End Reinforcement Learning Framework for Autonomous Navigation without Prior Knowledge

---

### Abstract
  >Real-world autonomous navigation demands the ability to operate in unknown, unstructured environments 
  using only onboard, limited-range sensor data, without reliance on prior maps or global information. 
  To address this fundamental challenge, this paper introduces a novel end-to-end navigation framework 
  that seamlessly integrates perception and planning. The core of our framework is Perception Net, an 
  interpretable, optimization-unfolded network that efficiently processes raw point clouds into a 
  compact representation of the immediate environment. This is coupled with a reinforcement learning 
  planner featuring a velocity-adaptive Bézier action space, which learns a sophisticated, reference-free 
  navigation policy. The agent's decisions are based solely on its own velocity, the relative goal 
  coordinates, and the real-time perception features. Extensive simulations demonstrate the framework's 
  high success rate and robust generalization to unseen non-convex scenarios. The practicality and 
  effectiveness of this map-less, purely reactive system are further validated through successful 
  real-world field tests on an Autonomous Surface Vehicle (ASV) platform, showcasing its capability for 
  truly autonomous navigation in complex environments. Future work will focus on handling dynamic 
  obstacles and improving robustness in escaping from extremely non-convex traps.

---

### Acknowledgments & License

This project is built upon several excellent open-source projects. We extend our sincere gratitude to their developers.

*   **[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3):** Our reinforcement learning agent is implemented using the robust and versatile Stable Baselines3 library.

*   **[NeuPAN](https://github.com/hanruihua/NeuPAN):** Our implementation and comparative analysis heavily reference and include modified code from the official NeuPAN repository. We thank the original authors for making their work public.

**License:** The original NeuPAN project is licensed under the **GNU General Public License v3.0**. In accordance with its terms, any derivative work must also be distributed under the same license. Therefore, this repository and all its contents are also licensed under the **GNU General Public License v3.0**. Please see the `LICENSE` file for more details.


---

## Introduction Video

[![PriorFree End-to-End Navigation Project Video](https://img.youtube.com/vi/3GgnUOSXJGo/0.jpg)](https://www.youtube.com/watch?v=3GgnUOSXJGo)

**Click the image above to watch our introduction video**

---
## Usage Guide

### 1. Clone the Repository

First, clone this repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/xidengwang/PriorFree_End2End_Navigation.git
cd PriorFree_End2End_Navigation
```

### 2. Environment Configuration

We provide two methods for setting up the Conda environment. Please choose one of the following options.

---

**Option A: Install via `requirements.txt`**

This method provides step-by-step control over the installation process.

1.  Create a new Conda environment with a specified Python version:
    ```bash
    conda create --name test_reqs_env python=3.12 -y
    ```

2.  Activate the newly created environment:
    ```bash
    conda activate test_reqs_env
    ```

3.  Install PyTorch, Torchvision, and Torchaudio compatible with your CUDA version. The command below is for CUDA 12.1:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    *Note: If you have a different CUDA version, please generate the correct installation command from the [official PyTorch website](https://pytorch.org/get-started/locally/).*

4.  Install the remaining dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

---

**Option B: Install via `environment.yml`**

This is the recommended and most straightforward method, as it automatically creates the complete environment from the provided file.

1.  Create the Conda environment using the `environment.yml` file. This command will install all necessary packages, including Python and PyTorch, with their specified versions.
    ```bash
    conda env create -f environment.yml
    ```
    *Note: The environment name (`end2end_env`) is defined within the `environment.yml` file.*

2.  Activate the environment:
    ```bash
    conda activate end2end_env
    ```

---

## Running Experiments

*Note: Ensure you are in the project's root directory (`PriorFree_End2End_Navigation`) and the correct Conda environment is activated before running any experiments.*

### Experiment 1: Perception Net vs. DUNE Comparison

This simulation is designed to compare the computational accuracy and speed of our proposed Perception Net against the DUNE model.

#### How to Run

Execute the following command from the project's root directory:

```bash
python -m experiments.01_perception_net_vs_dune.run_perception_comparison
```

Upon completion, the script will display the visualized comparison results directly as image plots.

#### Configuration

You can customize the test parameters by editing the script `experiments/01_perception_net_vs_dune/run_perception_comparison.py`:

-   **Accuracy Test**: To evaluate accuracy across different ranges, modify the `RANGE_SCALES` list, located around line 307.
-   **Speed Test**: To benchmark computation speed with different input sizes, modify the `SAMPLE_SIZES_SPEED` list, located around line 332.

> **Note:** For a reliable speed benchmark, it is crucial to run this test in a stable system environment with minimal background processes to ensure consistent and accurate measurements.

---

### Experiment 2: Comparison with DWA

This simulation compares the performance of our proposed RLplanner against the traditional Dynamic Window Approach (DWA) algorithm in various environments.

#### How to Run

Execute the following command from the project's root directory:

```bash
python -m experiments.02_comparison_with_dwa.run_dwa_comparison
```

The script will output result plots after the completion of each test environment.

#### Configuration

To modify the test environments or the number of test runs, edit the script `experiments/02_comparison_with_dwa/run_dwa_comparison.py`:

-   **Test Environments**: Adjust the `SEEDS_TO_TEST` list, located around line 121, to change the environments used for the comparison.

> **Note:** This simulation is computationally intensive and may take a significant amount of time to complete, as it runs tests across multiple environments sequentially.

---

### Experiment 3: Comparison with NRMP

This simulation provides a comparison between our RLplanner and the NRMP algorithm. To ensure a high-quality and consistent comparison, this test is conducted within a fixed, predefined environment.

#### How to Run

Execute the following command from the project's root directory:

```bash
python -m experiments.03_comparison_with_nrmp.run_nump_comparison
```

#### Configuration

You can adjust the parameters for this experiment as follows:

-   **NRMP Planner Parameters**: To fine-tune the NRMP algorithm's behavior, edit the configuration file located at `pretrained_models/NRMP_yaml/planner.yaml`.
-   **Test Environment**: To change the simulation environment, modify the `env_configs` parameters within the script `experiments/03_comparison_with_nrmp/run_nump_comparison.py`, located around lines 169-172.

> **Note:**
> -   The provided DUNE model for NRMP (representing the physical boat) was trained using code from the NeuPAN library.
> -   Its collision avoidance parameters were tuned in random environments, resulting in a conservative behavior profile.
> -   Consequently, for general environments without specific tuning, NRMP's performance may be limited. However, the algorithm can still perform exceptionally well if provided with a suitable reference path in advance.

---

### Experiment 4: Non-Convex Environment Test

This simulation is designed to evaluate the capability of our proposed RLplanner in navigating overlapping, non-convex environments.

#### How to Run

Execute the following command from the project's root directory:

```bash
python -m experiments.04_non_convex_environment.run_non_convex_test
```

#### Configuration

To customize the test environment, you can modify the parameters for the `eval_generator` in the script `experiments/04_non_convex_environment/run_non_convex_test.py`:

-   **Environment Parameters**: Adjust the configuration located around lines 126-134 to change the characteristics of the non-convex environment.

---

## Model Training

We are committed to open-sourcing the complete training pipeline. The code and instructions for training the models from scratch **will be made publicly available in this repository upon the official acceptance of our paper**.

In the meantime, we provide the pre-trained models used in our experiments to allow for the full reproduction of our evaluation results.

We appreciate your understanding and interest in our work. Please "watch" this repository to be notified of future updates, including the release of the training code.

## Physical Experiments

*To validate the effectiveness and robustness of our proposed method, we conducted a series of experiments in real-world physical environments. The video recordings of our six core experiments are available on YouTube via the links below.*

| No. | Experiment Description | Video Demo |
| :---: | :--- | :---: |
| **1** | **Navigation in Non-Convex Environments** <br>Navigating from the leftmost side to the rightmost side of the annular fan-shaped pool. | [Watch](https://youtu.be/2vQvciHE5nQ) |
| **2** | **Linear Navigation** <br> Navigating from the top-left to the far-right of the environment. | [Watch](https://youtu.be/dMHNBcFfxos) |
| **3** | **Reverse Navigation in Non-Convex Environments** <br> Navigating from the rightmost side to the leftmost side. | [Watch](https://youtu.be/JI68CFEDjp0) |
| **4** | **Right-Side Obstacle Avoidance** <br> Navigating around an obstacle positioned on the front-left. | [Watch](https://youtu.be/TYaJu90E2NU) |
| **5** | **Left-Side Obstacle Avoidance** <br> Navigating around an obstacle positioned on the front-right. | [Watch](https://youtu.be/QxsXPzGt7C8) |
| **6** | **Non-Convex Navigation with Obstacle Avoidance** <br> Navigating from the leftmost side to the rightmost side by bypassing an obstacle placed in the path. | [Watch](https://youtu.be/BlnDSGzK1CI) |