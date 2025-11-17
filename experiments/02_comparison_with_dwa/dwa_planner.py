# --- START OF FILE experiments/02_comparison_with_dwa/dwa_planner.py ---
"""
Implementation of the Dynamic Window Approach (DWA) planner.
"""

import math
import numpy as np
from numba import njit
from typing import List, Tuple

# Forward declaration for type hinting to avoid potential circular imports
if False:
    from ship_rl_planner import OptimizedPerceptionModule


class DWAConfig:
    """
    Configuration parameters for the DWA planner.
    """

    def __init__(self):
        """Initializes the DWA configuration with default values."""
        # Vehicle dynamics parameters
        self.max_speed: float = 5.0  # m/s
        self.min_speed: float = 0.0  # m/s
        self.max_yaw_rate: float = np.deg2rad(45.0)  # rad/s
        self.max_accel: float = 2.0  # m/s^2
        self.max_delta_yaw_rate: float = np.deg2rad(100.0)  # rad/s^2

        # Sampling resolution for the dynamic window
        self.v_resolution: float = 0.2  # m/s
        self.yaw_rate_resolution: float = np.deg2rad(2.0)  # rad/s

        # Prediction and simulation parameters
        self.dt: float = 0.1  # s
        self.predict_time: float = 3.0  # s

        # Cost function weights
        self.goal_cost_gain: float = 0.15
        self.speed_cost_gain: float = 1.0
        self.obstacle_cost_gain: float = 1.0

        # Safety parameters
        self.robot_radius: float = 0.8  # meters, safety margin for collision


@njit(cache=True)
def motion_model_numba(state: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    """Simulates one step of the vehicle's motion using a kinematic model."""
    x, y, yaw, _, _ = state
    
    # Convert navigation yaw (Y-fwd=0) to standard math yaw (X-fwd=0)
    math_yaw = -yaw + np.pi / 2
    
    new_x = x + v * math.cos(math_yaw) * dt
    new_y = y + v * math.sin(math_yaw) * dt
    new_yaw = yaw + w * dt
    
    # Normalize angle to the range [-pi, pi]
    if new_yaw > np.pi:
        new_yaw -= 2.0 * np.pi
    elif new_yaw < -np.pi:
        new_yaw += 2.0 * np.pi
        
    return np.array([new_x, new_y, new_yaw, v, w])


@njit(cache=True)
def predict_trajectory_numba(state_init: np.ndarray, v: float, w: float, dt: float, predict_time: float) -> np.ndarray:
    """Generates a future trajectory by repeatedly applying the motion model."""
    state = np.copy(state_init)
    num_steps = int(predict_time / dt)
    
    trajectory = np.empty((num_steps + 1, 5), dtype=np.float64)
    trajectory[0] = state

    for i in range(num_steps):
        state = motion_model_numba(state, v, w, dt)
        trajectory[i + 1] = state
        
    return trajectory


class DWAPlanner:
    """
    A high-performance DWA planner with batch-based collision checking.
    """

    def __init__(self, config: DWAConfig, perception_module: 'OptimizedPerceptionModule'):
        """Initializes the DWA planner."""
        self.config = config
        self.perception_module = perception_module

    def plan(self, state: np.ndarray, goal: np.ndarray, pointcloud_body: np.ndarray) -> List[float]:
        """Calculates the best control command (v, w) using the DWA algorithm."""
        dynamic_window = self._calc_dynamic_window(state)
        best_cost = float("inf")
        best_control = [0.0, 0.0]

        # 1. Generate all candidate trajectories
        trajectories = []
        control_params = []
        v_samples = np.arange(dynamic_window[0], dynamic_window[1] + self.config.v_resolution, self.config.v_resolution)
        w_samples = np.arange(dynamic_window[2], dynamic_window[3] + self.config.yaw_rate_resolution, self.config.yaw_rate_resolution)
        
        if len(v_samples) == 0 or len(w_samples) == 0:
            return best_control

        for v in v_samples:
            for w in w_samples:
                trajectory = predict_trajectory_numba(state, v, w, self.config.dt, self.config.predict_time)
                trajectories.append(trajectory)
                control_params.append((v, w))

        if not trajectories:
            return best_control

        # 2. Batch evaluate the safety of all trajectories
        obstacle_costs = self._batch_evaluate_obstacle_cost(
            trajectories, pointcloud_body, state[0:2], state[2]
        )

        # 3. Iterate through trajectories to find the one with the minimum total cost
        for i, trajectory in enumerate(trajectories):
            if obstacle_costs[i] == float("inf"):
                continue

            dx = goal[0] - trajectory[-1, 0]
            dy = goal[1] - trajectory[-1, 1]
            goal_cost = math.hypot(dx, dy)
            speed_cost = self.config.max_speed - trajectory[-1, 3]

            final_cost = (self.config.goal_cost_gain * goal_cost +
                          self.config.speed_cost_gain * speed_cost +
                          self.config.obstacle_cost_gain * obstacle_costs[i])

            if final_cost < best_cost:
                best_cost = final_cost
                best_control = [control_params[i][0], control_params[i][1]]

        return best_control

    def _batch_evaluate_obstacle_cost(self, trajectories: List[np.ndarray], pointcloud_body: np.ndarray, current_pos: np.ndarray, current_yaw: float) -> np.ndarray:
        """Evaluates the obstacle cost for a list of trajectories in a single batch."""
        if len(pointcloud_body) == 0:
            return np.zeros(len(trajectories))

        pointcloud_world = self._body_to_world(pointcloud_body, current_pos, current_yaw)

        points_to_transform = []
        traj_point_indices = [0]
        
        for trajectory in trajectories:
            for point_on_traj in trajectory:
                points_to_transform.append(point_on_traj)
            traj_point_indices.append(len(points_to_transform))

        points_to_transform = np.array(points_to_transform)
        
        all_transformed_points = self._batch_world_to_body(
            pointcloud_world, points_to_transform[:, 0:2], points_to_transform[:, 2]
        )

        batch_distances = self.perception_module.predict_distance_batch(all_transformed_points)
        
        obstacle_costs = np.zeros(len(trajectories))
        for i in range(len(trajectories)):
            start_idx = traj_point_indices[i] * len(pointcloud_body)
            end_idx = traj_point_indices[i+1] * len(pointcloud_body)
            
            traj_distances = batch_distances[start_idx:end_idx]
            
            min_dist = np.min(traj_distances) if len(traj_distances) > 0 else float('inf')
            
            if min_dist <= self.config.robot_radius:
                obstacle_costs[i] = float("inf")
            else:
                obstacle_costs[i] = 1.0 / min_dist

        return obstacle_costs
        
    def _calc_dynamic_window(self, state: np.ndarray) -> List[float]:
        """Calculates the dynamic window of achievable velocities."""
        v_static = [self.config.min_speed, self.config.max_speed,
                    -self.config.max_yaw_rate, self.config.max_yaw_rate]
        
        v_dynamic = [state[3] - self.config.max_accel * self.config.dt,
                     state[3] + self.config.max_accel * self.config.dt,
                     state[4] - self.config.max_delta_yaw_rate * self.config.dt,
                     state[4] + self.config.max_delta_yaw_rate * self.config.dt]
                     
        return [max(v_static[0], v_dynamic[0]), min(v_static[1], v_dynamic[1]),
                max(v_static[2], v_dynamic[2]), min(v_static[3], v_dynamic[3])]

    def _body_to_world(self, points_body: np.ndarray, ship_pos: np.ndarray, ship_yaw: float) -> np.ndarray:
        """Transforms a batch of points from the body frame to the world frame."""
        cos_y, sin_y = np.cos(ship_yaw), np.sin(ship_yaw)
        rot_matrix = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        return (rot_matrix @ points_body.T).T + ship_pos
    
    def _batch_world_to_body(self, points_world_batch: np.ndarray, ship_poses: np.ndarray, ship_yaws: np.ndarray) -> np.ndarray:
        """Transforms a single point cloud to multiple body frames defined by ship poses."""
        rel_points = points_world_batch[np.newaxis, :, :] - ship_poses[:, np.newaxis, :]
        
        cos_yaws = np.cos(ship_yaws)
        sin_yaws = np.sin(ship_yaws)
        
        inv_rot_matrices = np.array([
            [cos_yaws, sin_yaws],
            [-sin_yaws, cos_yaws]
        ]).transpose(2, 0, 1)

        points_body = np.einsum('pij,pkj->pki', inv_rot_matrices, rel_points)
        
        return points_body.reshape(-1, 2)

# --- END OF FILE experiments/02_comparison_with_dwa/dwa_planner.py ---