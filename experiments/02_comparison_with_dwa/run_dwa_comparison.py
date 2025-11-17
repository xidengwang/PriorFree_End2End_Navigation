# --- START OF FILE experiments/02_comparison_with_dwa/run_dwa_comparison.py ---
"""
Main script for conducting a comparative evaluation between the trained RL agent
and the DWA (Dynamic Window Approach) planner.

This script executes the following steps for a range of seeds:
1.  Generates a deterministic environment using the `fixed_env` generator.
2.  Runs a simulation with the DWA planner on this environment.
3.  Runs a simulation with the pre-trained SAC RL agent on the identical environment.
4.  Calculates key performance metrics (success rate, steps, trajectory curvature,
    planning time) for both planners.
5.  Saves the results, including trajectory plots and log files, to the `outputs`
    directory for analysis.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from stable_baselines3 import SAC

from ship_rl_planner import (
    fixed_env,
    MyShipEnv,
    OptimizedPerceptionModule,
    LidarScanner,
)
from .dwa_planner import DWAConfig, DWAPlanner, motion_model_numba

ROBOT_G = np.array([[1, 0], [-1, 0], [0, -1], [2, 1], [-2, 1]], dtype=np.float32)
ROBOT_B = np.array([[0.25], [0.25], [0.4], [0.7], [0.7]], dtype=np.float32)


def setup_directories_and_files(output_dir: str, img_subdir: str) -> Tuple[str, str]:
    """Checks and creates output directories and log files."""
    img_dir = os.path.join(output_dir, img_subdir)
    os.makedirs(img_dir, exist_ok=True)
    # Clear log files at the beginning of a run
    with open(os.path.join(output_dir, "RL.txt"), 'w'): pass
    with open(os.path.join(output_dir, "DWA.txt"), 'w'): pass
    return output_dir, img_dir

def calculate_average_curvature(trajectory_points: List[np.ndarray]) -> float:
    """Calculates the average curvature of a trajectory using a numerical method."""
    if len(trajectory_points) < 3: return 0.0
    traj = np.array(trajectory_points)
    dx, dy = np.gradient(traj[:, 0]), np.gradient(traj[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5
    valid_indices = denominator > 1e-6
    if not np.any(valid_indices): return 0.0
    return np.mean(numerator[valid_indices] / denominator[valid_indices])

def plot_and_save_results(obstacles: List[np.ndarray], goal: np.ndarray, trajectory: List[np.ndarray],
                          method_name: str, seed: int, result_status: str, save_path: str):
    """Generates and saves a plot of the simulation results with consistent styling."""
    trajectory = np.array(trajectory)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("#e0e0e0")

    # Draw obstacles
    for obs in obstacles:
        ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.6))

    # Draw goal area
    ax.add_patch(plt.Circle(goal, 5.0, facecolor='green', alpha=0.3, edgecolor='darkgreen',
                             linestyle='--', linewidth=2, label='Goal Area'))
    
    # Draw trajectory with specified style
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'g--', linewidth=2, label=f'{method_name} Trajectory')
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'yo', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'go', markersize=10, label='End')
    
    # Set titles and labels (title is commented out to match the reference)
    # title = f"{method_name} | Seed: {seed} | Result: {result_status}"
    # ax.set_title(title)
    ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)")
    ax.set_xlim(0, 200), ax.set_ylim(0, 200)
    ax.set_aspect('equal', adjustable='box'), ax.legend(loc='lower right'), ax.grid(True, linestyle=':', color='white')
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # --- Setup ---
    # Note: Path is relative to the project root where the script is likely run from.
    OUTPUT_DIR, IMG_DIR = setup_directories_and_files(output_dir="experiments/02_comparison_with_dwa/outputs", img_subdir="images")
    RL_LOG_FILE = os.path.join(OUTPUT_DIR, "RL.txt")
    DWA_LOG_FILE = os.path.join(OUTPUT_DIR, "DWA.txt")
    
    # Correctly locate models relative to the project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RL_MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_models/rl_agent/sac_ship_planner_final.zip")
    PERCEPTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_models/rl_agent/perception_net_ship.pth")
    
    if not os.path.exists(RL_MODEL_PATH):
        raise FileNotFoundError(f"RL model not found at: {RL_MODEL_PATH}")
    if not os.path.exists(PERCEPTION_MODEL_PATH):
        raise FileNotFoundError(f"Perception model not found at: {PERCEPTION_MODEL_PATH}")

    print("Loading SAC model...")
    rl_model = SAC.load(RL_MODEL_PATH, device='cuda')
    
    print("Initializing Perception Module for DWA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perception_module_for_dwa = OptimizedPerceptionModule(model_path=PERCEPTION_MODEL_PATH, G=ROBOT_G, b=ROBOT_B, device=device)
    
    print("Initializing DWA planner...")
    dwa_config = DWAConfig()
    dwa_planner = DWAPlanner(dwa_config, perception_module_for_dwa)

    # --- Simulation Parameters ---
    SEEDS_TO_TEST = range(100)
    MAX_SIM_TIME = 80.0
    DT = 0.1
    MAX_STEPS = int(MAX_SIM_TIME / DT)
    ENV_CONFIG = {'bezier_n': 5, 'band_width': 6, 'obstacle_density': 0.24, 'x_variation_amplitude': 30.0}

    for seed in tqdm(SEEDS_TO_TEST, desc="Evaluating Seeds"):
        
        # --- RL Agent Evaluation ---
        rl_env_generator = fixed_env(random_seed=seed, **ENV_CONFIG)
        rl_env = MyShipEnv(
            perception_model_path=PERCEPTION_MODEL_PATH,
            env_generator=rl_env_generator
        )
        obs, info = rl_env.reset()
        
        trajectory_rl = [rl_env.ship_state[:2].copy()]
        times_rl = []
        rl_result = "Timeout"
        done, truncated = False, False
        
        while not done and not truncated:
            start_time = time.perf_counter()
            action, _ = rl_model.predict(obs, deterministic=True)
            times_rl.append((time.perf_counter() - start_time) * 1000)
            
            obs, _, done, truncated, info = rl_env.step(action)
            trajectory_rl.append(rl_env.ship_state[:2].copy())
            

        if info.get('is_success', False): rl_result = "Success"
        elif info.get('collided', False): rl_result = "Collision"
        
        avg_time_rl = np.mean(times_rl) if times_rl else 0
        avg_curvature_rl = calculate_average_curvature(trajectory_rl)
        plot_and_save_results(rl_env.obstacles, rl_env.current_episode_goal_world, trajectory_rl, "RL", seed, rl_result,
                              os.path.join(IMG_DIR, f"RL_{seed}.png"))
        
        if rl_result == "Success":
            with open(RL_LOG_FILE, 'a') as f:
                f.write(f"{seed} {len(trajectory_rl)-1} {avg_curvature_rl:.6f} {avg_time_rl:.6f}\n")
        rl_env.close()

        # --- DWA Planner Evaluation ---
        dwa_env_generator = fixed_env(random_seed=seed, **ENV_CONFIG)
        env_config_dict = dwa_env_generator.generate_environment()
        obstacles_dwa = [np.array(o) for o in env_config_dict["obstacles"]]
        goal_dwa = np.array(env_config_dict["target_point"])
        
        current_state_dwa = np.array([100.0, 5.0, 0.0, 0.0, 0.0]) # x, y, yaw, v, w
        lidar_scanner_dwa = LidarScanner() # Does not need perception module for point cloud generation
        trajectory_dwa = [current_state_dwa[:2].copy()]
        times_dwa = []; dwa_result = "Timeout"
        
        for step in range(MAX_STEPS):
            ship_pos, ship_yaw = current_state_dwa[:2], current_state_dwa[2]
            pointcloud, _ = lidar_scanner_dwa.scan_environment(ship_pos, ship_yaw, obstacles_dwa)
            
            start_time = time.perf_counter()
            u = dwa_planner.plan(current_state_dwa, goal_dwa, pointcloud)
            times_dwa.append((time.perf_counter() - start_time) * 1000)

            last_state = np.copy(current_state_dwa)
            current_state_dwa = motion_model_numba(current_state_dwa, u[0], u[1], DT)
            trajectory_dwa.append(current_state_dwa[:2].copy())

            # Collision check using the same LidarScanner logic
            _, dists = lidar_scanner_dwa.scan_environment(current_state_dwa[:2], current_state_dwa[2], obstacles_dwa)
            if dists.size > 0 and np.min(dists) < 0.01: dwa_result = "Collision"; break
            if np.linalg.norm(current_state_dwa[:2] - goal_dwa) < 5.0: dwa_result = "Success"; break
            if np.allclose(current_state_dwa[:2], last_state[:2], atol=1e-4): break # Timeout if stuck
        
        avg_time_dwa = np.mean(times_dwa) if times_dwa else 0
        avg_curvature_dwa = calculate_average_curvature(trajectory_dwa)
        plot_and_save_results(obstacles_dwa, goal_dwa, trajectory_dwa, "DWA", seed, dwa_result,
                              os.path.join(IMG_DIR, f"DWA_{seed}.png"))
        
        if dwa_result == "Success":
            with open(DWA_LOG_FILE, 'a') as f:
                f.write(f"{seed} {len(trajectory_dwa)-1} {avg_curvature_dwa:.6f} {avg_time_dwa:.6f}\n")

    print("\n\n" + "="*45)
    print("All evaluations completed!")
    print(f"Results saved in '{OUTPUT_DIR}' directory.")
    print("="*45)

# --- END OF FILE experiments/02_comparison_with_dwa/run_dwa_comparison.py ---