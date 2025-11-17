# --- START OF FILE experiments/03_comparison_with_nrmp/run_nrmp_comparison.py ---
"""
Runs a side-by-side comparison of the NRMP (NeuPAN) planner and the trained
RL agent on two predefined, fixed environments.
This script is self-contained and handles Python path issues to ensure it can be
run directly.
"""

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_NEUPAN_PATH = os.path.join(_PROJECT_ROOT, 'dependencies', 'NeuPAN')
if _NEUPAN_PATH not in sys.path:
    sys.path.insert(0, _NEUPAN_PATH)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from stable_baselines3 import SAC

from neupan import neupan
from ship_rl_planner import MyShipEnv, fixed_env, LidarScanner
from tqdm import tqdm


class PredefinedEnvGenerator:
    """A helper class to force MyShipEnv to use a pre-generated environment."""
    def __init__(self, environment_data: dict):
        self.env_data = environment_data
    
    def generate_environment(self):
        return self.env_data

def run_nrmp_simulation(environment_data: dict, config_path: str) -> np.ndarray:
    """Runs NRMP simulation replicating the EXACT logic of the original script."""
    print("  - Running NRMP (NeuPAN) simulation...")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"FATAL: NeuPAN config file 'planner.yaml' not found at: {config_path}.")
    planner = neupan.init_from_yaml(config_path)
    
    obstacles_list = [np.array(obs) for obs in environment_data["obstacles"]]
    goal_point = np.array(environment_data["target_point"])
    
    initial_state = np.array([[100.0], [5.0], [np.pi/2]])
    goal_state = np.array([[goal_point[0]], [goal_point[1]], [np.pi/2]])
    
    current_state = initial_state.copy()
    planner.update_initial_path_from_waypoints([initial_state, goal_state])
    planner.reset()
    
    lidar = LidarScanner()
    dt = planner.dt
    max_steps = 2000
    trajectory = [current_state.flatten()]
    
    for _ in range(max_steps):
        current_pos = current_state[:2, 0]
        math_yaw = current_state[2, 0]
        _, scan_distances = lidar.scan_environment(current_pos, math_yaw, obstacles_list)

        scan_data_for_neupan = {
            'ranges': scan_distances, 'angle_min': -np.pi, 'angle_max': np.pi,
            'range_min': 0.0, 'range_max': lidar.max_range,
        }
        
        obstacle_points_world = planner.scan_to_point(state=current_state, scan=scan_data_for_neupan)
        action, info = planner.forward(state=current_state, points=obstacle_points_world, velocities=None)
        
        v, omega = action[0, 0], action[1, 0]
        current_state[0] += v * np.cos(current_state[2, 0]) * dt
        current_state[1] += v * np.sin(current_state[2, 0]) * dt
        current_state[2] += omega * dt
        
        trajectory.append(current_state.flatten())
        
        if info['arrive'] or info['stop'] or info['collision']:
            break
            
    return np.array(trajectory)

def run_rl_simulation(environment_data: dict, rl_model: SAC, perception_model_path: str) -> np.ndarray:
    """Runs RL sim and returns the FULL trajectory [x, y, vx, vy, nav_yaw]."""
    print("  - Running RL (SAC) simulation...")
    predefined_generator = PredefinedEnvGenerator(environment_data)
    rl_env = MyShipEnv(perception_model_path=perception_model_path, env_generator=predefined_generator)
    obs, _ = rl_env.reset()
    trajectory = [rl_env.ship_state.copy()]
    done, truncated = False, False    
        
    with tqdm(total=rl_env.max_episode_steps, desc=f"Simulating Ep {i+1}") as pbar:
        while not done and not truncated:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = rl_env.step(action)
            trajectory.append(rl_env.ship_state.copy())
            pbar.update(1)

    rl_env.close()
    return np.array(trajectory)

START_POINT = np.array([100.0, 5.0])

def plot_single_comparison(ax, plot_data: dict):
    """Draws the comparison plot, including detailed ship poses."""
    nrmp_full_traj, rl_full_traj = plot_data['nrmp_traj'], plot_data['rl_traj']
    obstacles, goal, goal_threshold = plot_data['obstacles'], plot_data['goal'], plot_data['goal_threshold']
    
    ax.set_facecolor("#e0e0e0")

    for obs in obstacles:
        ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.6))

    ax.add_patch(plt.Circle(goal, goal_threshold, facecolor='green', alpha=0.3,
                           edgecolor='darkgreen', linestyle='--', linewidth=2))
    
    ship_vertices_body = np.array([[-0.25, -0.40], [-0.25,  0.20], [ 0.00,  0.70], [ 0.25,  0.20], [ 0.25, -0.40]])
    
    if len(nrmp_full_traj) > 0:
        for i, state in enumerate(nrmp_full_traj):
            if i % 2 == 0:
                x, y, theta = state
                rot_matrix = np.array([[np.cos(theta - np.pi/2), -np.sin(theta - np.pi/2)], [np.sin(theta - np.pi/2),  np.cos(theta - np.pi/2)]])
                ship_vertices_world = (rot_matrix @ ship_vertices_body.T).T + np.array([x, y])
                ax.plot(np.vstack([ship_vertices_world, ship_vertices_world[0]])[:, 0], np.vstack([ship_vertices_world, ship_vertices_world[0]])[:, 1], color='darkviolet', alpha=0.5, linewidth=0.5)
        ax.plot(nrmp_full_traj[:, 0], nrmp_full_traj[:, 1], '--', color='darkviolet', linewidth=2.0, label='NRMP Trajectory')

    if len(rl_full_traj) > 0:
        for i, state in enumerate(rl_full_traj):
            if i % 2 == 0:
                x, y, _, _, nav_yaw = state
                rot_matrix = np.array([[np.cos(nav_yaw), -np.sin(nav_yaw)], [np.sin(nav_yaw),  np.cos(nav_yaw)]])
                ship_vertices_world = (rot_matrix @ ship_vertices_body.T).T + np.array([x, y])
                ax.plot(np.vstack([ship_vertices_world, ship_vertices_world[0]])[:, 0], np.vstack([ship_vertices_world, ship_vertices_world[0]])[:, 1], color='green', alpha=0.5, linewidth=0.5)
        ax.plot(rl_full_traj[:, 0], rl_full_traj[:, 1], '-', color='green', linewidth=2.0, label='RL Trajectory')

    ax.plot(START_POINT[0], START_POINT[1], 'yo', markersize=10, markeredgecolor='k', label='Start Point')
    if len(nrmp_full_traj) > 1: ax.plot(nrmp_full_traj[-1, 0], nrmp_full_traj[-1, 1], 'o', color='darkviolet', markersize=10, markeredgecolor='k')
    if len(rl_full_traj) > 1: ax.plot(rl_full_traj[-1, 0], rl_full_traj[-1, 1], 'o', color='green', markersize=10, markeredgecolor='k')
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', color='white')

if __name__ == '__main__':
    RL_MODEL_PATH = os.path.join(_PROJECT_ROOT, "pretrained_models/rl_agent/sac_ship_planner_final.zip")
    PERCEPTION_MODEL_PATH = os.path.join(_PROJECT_ROOT, "pretrained_models/rl_agent/perception_net_ship.pth")
    NEUPAN_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'pretrained_models/NRMP_yaml/planner.yaml')
    OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "experiments/03_comparison_with_nrmp/outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(RL_MODEL_PATH):
        raise FileNotFoundError(f"RL model not found at: {RL_MODEL_PATH}")

    print("Loading SAC model...")
    rl_model = SAC.load(RL_MODEL_PATH)
    print("SAC model loaded successfully.")

    env_configs = {
        "env1": {'random_seed': 7, 'bezier_n': 3, 'band_width': 10.0, 'obstacle_density': 0.06, 'x_variation_amplitude': 10},
        "env2": {'random_seed': 7, 'bezier_n': 4, 'band_width': 8.0, 'obstacle_density': 0.2, 'x_variation_amplitude': 20}
    }

    fig, axes = plt.subplots(1, 2, figsize=(22, 12))

    for i, (env_name, config) in enumerate(env_configs.items()):
        print(f"\n[Processing Scenario: {env_name.upper()}]")
        master_generator = fixed_env(**config)
        shared_environment_data = master_generator.generate_environment()
        
        nrmp_full_trajectory = run_nrmp_simulation(shared_environment_data, NEUPAN_CONFIG_PATH)
        rl_full_trajectory = run_rl_simulation(shared_environment_data, rl_model, PERCEPTION_MODEL_PATH)

        plot_data = {
            'nrmp_traj': nrmp_full_trajectory,
            'rl_traj': rl_full_trajectory,
            'obstacles': shared_environment_data["obstacles"],
            'goal': np.array(shared_environment_data["target_point"]),
            'goal_threshold': 5.0
        }
        
        plot_single_comparison(axes[i], plot_data)
        axes[i].legend()

    print("\nSimulations complete. Finalizing plot...")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=20, bbox_to_anchor=(0.5, 1), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = os.path.join(OUTPUT_DIR, "NRMP_vs_RL_comparison.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"\nComparison plot saved successfully as '{output_path}'")

# --- END OF FILE experiments/03_comparison_with_nrmp/run_nrmp_comparison.py ---