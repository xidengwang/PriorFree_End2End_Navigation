# --- START OF FILE experiments/04_non_convex_environment/run_non_convex_test.py ---
"""
Evaluates the pre-trained RL agent in challenging non-convex environments
where obstacles are allowed to overlap, creating complex shapes.

This script performs the following steps:
1.  Defines a custom environment generator that allows obstacles to overlap.
2.  Loads the final pre-trained SAC agent.
3.  Runs a series of evaluation episodes in these complex environments.
4.  For each successful episode, saves a plot of the generated trajectory.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import SAC

# --- Robust Path Setup ---
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# --- End of Path Setup ---

from ship_rl_planner import MyShipEnv, ObstacleEnvironmentGenerator, BezierCurveUtils


class OverlappingObstacleGenerator(ObstacleEnvironmentGenerator):
    """
    A specialized generator that inherits from the standard one but removes
    the collision check between obstacles, thus allowing them to overlap and
    form non-convex shapes.
    """
    def _generate_obstacles_outside_band(self, num_attempts: int = 1500) -> list[np.ndarray]:
        obstacles = []
        path_samples = np.array([
            BezierCurveUtils.bezier_point(self.internal_reference_control_points, t)
            for t in np.linspace(0, 1, 100)
        ])

        world_area = (self.world_bounds_x[1] - self.world_bounds_x[0]) * (self.world_bounds_y[1] - self.world_bounds_y[0])
        _, _, total_path_length = BezierCurveUtils.create_arc_length_lookup_table(self.internal_reference_control_points)
        band_area_estimate = total_path_length * self.band_width
        available_area = max(0, world_area - band_area_estimate)
        
        avg_radius = (self.min_obs_radius + self.max_obs_radius) / 2
        avg_obs_area = np.pi * avg_radius**2 if avg_radius > 0 else 0
        expected_num_obstacles = int((available_area * self.obstacle_density) / avg_obs_area) if avg_obs_area > 0 else 0

        for _ in range(num_attempts):
            if len(obstacles) >= expected_num_obstacles * 2 and len(obstacles) > 10:
                break

            obs_x = self.np_random.uniform(self.world_bounds_x[0], self.world_bounds_x[1])
            obs_y = self.np_random.uniform(self.world_bounds_y[0], self.world_bounds_y[1])
            obs_radius = self.np_random.uniform(self.min_obs_radius, self.max_obs_radius)
            new_obs_center = np.array([obs_x, obs_y])

            min_dist_to_path = np.min(np.linalg.norm(path_samples - new_obs_center, axis=1))
            if min_dist_to_path < (self.band_width / 2 + obs_radius):
                continue

            if np.linalg.norm(new_obs_center - self.initial_point) < (obs_radius + 10.0):
                continue
            if np.linalg.norm(new_obs_center - self.target_point) < (obs_radius + 10.0):
                continue

            obstacles.append(np.array([obs_x, obs_y, obs_radius], dtype=np.float32))
            
        return obstacles


def plot_trajectory_simple(env: MyShipEnv, trajectory_states: list, save_path: str):
    """
    Visualizes an episode trajectory by plotting only the centroid path as a
    green dashed line, without drawing individual ship poses.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("#e0e0e0")

    if hasattr(env, 'obstacles') and env.obstacles:
        for obs in env.obstacles:
            ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color='#F35A5A', alpha=1))
    
    if hasattr(env, 'current_episode_goal_world') and env.current_episode_goal_world is not None:
        goal_pos = env.current_episode_goal_world
        ax.add_patch(plt.Circle(goal_pos, env.goal_threshold, 
                                     facecolor='green', alpha=0.3, 
                                     edgecolor='darkgreen', linestyle='--',
                                     linewidth=2, label='Goal Area'))

    if trajectory_states:
        positions = np.array([state[:2] for state in trajectory_states])
        ax.plot(positions[:, 0], positions[:, 1], 'g--', linewidth=2, label='Ship Trajectory')
        ax.plot(positions[0, 0], positions[0, 1], 'yo', markersize=8, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'bo', markersize=8, label='End')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim((0,200))
    ax.set_ylim((0,200))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', color='white')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    RL_MODEL_PATH = os.path.join(_PROJECT_ROOT, "pretrained_models/rl_agent/sac_ship_planner_final.zip")
    PERCEPTION_MODEL_PATH = os.path.join(_PROJECT_ROOT, "pretrained_models/rl_agent/perception_net_ship.pth")
    OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "experiments/04_non_convex_environment/outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    NUM_EVAL_EPISODES = 100

    print("--- Starting evaluation in non-convex (overlapping) environments ---")

    if not os.path.exists(RL_MODEL_PATH):
        raise FileNotFoundError(f"FATAL: Model file not found at '{RL_MODEL_PATH}'!")

    eval_generator = OverlappingObstacleGenerator(
        bezier_n=3,
        band_width=10,
        obstacle_density=0.06,
        x_variation_amplitude=10,
        min_obs_radius=1.0,
        max_obs_radius=15.0,
        verbose=False
    )
    
    eval_env = MyShipEnv(
        perception_model_path=PERCEPTION_MODEL_PATH,
        env_generator=eval_generator,
        max_collisions_per_env=1
    )

    print(f"Loading SAC model from '{RL_MODEL_PATH}'...")
    model = SAC.load(RL_MODEL_PATH, env=eval_env)
    print("Model loaded successfully.")

    for i in range(NUM_EVAL_EPISODES):
        print(f"\n>>> Evaluating Episode {i + 1}/{NUM_EVAL_EPISODES}")
        
        obs, _ = eval_env.reset()
        done, truncated = False, False
        
        ship_trajectory_states = [eval_env.ship_state.copy()]

        with tqdm(total=eval_env.max_episode_steps, desc=f"Simulating Ep {i+1}") as pbar:
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                ship_trajectory_states.append(eval_env.ship_state.copy())
                pbar.update(1)
        
        result_status = "Timeout"
        if info.get('is_success', False): result_status = "Success"
        elif info.get('collided', False): result_status = "Collision"
        
        print(f"Episode {i + 1} finished.")
        print(f"  - Result: {result_status}")
        print(f"  - Total Steps: {eval_env.current_step}")

        if info.get('is_success', False):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"overlapping_traj_ep{i+1}_{timestamp}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                print(f"  - Success! Saving trajectory plot to '{filepath}'")
                plot_trajectory_simple(
                    env=eval_env,
                    trajectory_states=ship_trajectory_states,
                    save_path=filepath
                )
            except Exception as e:
                print(f"  - Error saving plot: {e}")
            
    eval_env.close()
    print("\n--- Evaluation complete ---")

# --- END OF FILE experiments/04_non_convex_environment/run_non_convex_test.py ---