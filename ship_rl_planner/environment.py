# --- START OF FILE ship_rl_planner/environment.py ---
"""
Core components for the USV (Unmanaged Surface Vessel) reinforcement learning environment.

This file defines:
1.  ObstacleEnvironmentGenerator: A class to procedurally generate random environments
    with obstacles for training and evaluation.
2.  MyShipEnv: The main Gymnasium-compatible environment for the USV navigation task,
    integrating dynamics, perception, and reward functions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

from .perception import OptimizedPerceptionModule
from .utils.bezier import BezierCurveUtils
from .utils.lidar import LidarScanner


# ==================== Environment Constants ====================
BEZIER_N: int = 4
ALPHA_I: Dict[int, float] = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
RL_PLANNING_INTERVAL_DELTA_T: float = 0.1
LOW_LEVEL_CONTROL_DT: float = 0.01

THRESHOLD_GOAL_DIST: float = 5.0
MAX_EPISODE_STEPS: int = 800
COLLISION_THRESHOLD: float = 0.01
PATH_SAMPLE_POINTS_M: int = 50

REWARD_GOAL_REACHED: float = 1000.0
REWARD_COLLISION: float = -800.0
EPSILON_1: float = 10000.0
EPSILON_2: float = 500.0
EPSILON_3: float = 3.0
EPSILON_4: float = 0.5
EPSILON_5: float = 4.0

V_MIN, V_MAX = 0.0, 5.0
A_MAX: float = 2.0

ROBOT_G: np.ndarray = np.array([[1, 0], [-1, 0], [0, -1], [2, 1], [-2, 1]], dtype=np.float32)
ROBOT_B: np.ndarray = np.array([[0.25], [0.25], [0.4], [0.7], [0.7]], dtype=np.float32)


# ==================== Environment Generator ====================

class ObstacleEnvironmentGenerator:
    """
    Generates environments with a random reference path (Bezier curve) and
    randomly placed obstacles that avoid this path.
    """

    def __init__(self,
                 random_seed: Optional[int] = None,
                 bezier_n: int = 5,
                 band_width: float = 6.0,
                 obstacle_density: float = 0.24,
                 min_obs_radius: float = 1.0,
                 max_obs_radius: float = 10.0,
                 x_variation_amplitude: float = 30.0,
                 world_bounds_x: Tuple[float, float] = (0.0, 200.0),
                 world_bounds_y: Tuple[float, float] = (0.0, 200.0),
                 verbose: bool = False):
        self.np_random = np.random.default_rng(random_seed)
        self._seed_counter = 0
        self.bezier_n = max(2, bezier_n)
        self.band_width = band_width
        self.obstacle_density = obstacle_density
        self.min_obs_radius = min_obs_radius
        self.max_obs_radius = max_obs_radius
        self.x_variation_amplitude = x_variation_amplitude
        self.world_bounds_x = world_bounds_x
        self.world_bounds_y = world_bounds_y
        self.verbose = verbose
        self.initial_point: np.ndarray = np.array([100.0, 5.0], dtype=np.float32)
        self.target_point: Optional[np.ndarray] = None
        self.internal_reference_control_points: List[List[float]] = []
        self.generated_obstacles: List[np.ndarray] = []

    def _update_random_state(self) -> None:
        self._seed_counter += 1
        new_seed = int(time.time() * 1000000) % 2**32 + self._seed_counter
        self.np_random = np.random.default_rng(new_seed)

    def _generate_random_target_point(self) -> np.ndarray:
        goal_x_min = self.world_bounds_x[0] + 5.0
        goal_x_max = self.world_bounds_x[1] - 5.0
        goal_y = 190.0

        if goal_x_min >= goal_x_max:
            goal_x = (self.world_bounds_x[0] + self.world_bounds_x[1]) / 2.0
        else:
            goal_x = self.np_random.uniform(goal_x_min, goal_x_max)

        return np.array([goal_x, goal_y], dtype=np.float32)

    def _generate_internal_reference_bezier_control_points(self) -> List[List[float]]:
        control_points = [self.initial_point.tolist()]
        min_x_world_padded = self.world_bounds_x[0] + 5
        max_x_world_padded = self.world_bounds_x[1] - 5
        min_y_world_padded = self.world_bounds_y[0] + 5
        max_y_world_padded = self.world_bounds_y[1] - 5

        for i in range(1, self.bezier_n):
            py_val = self.target_point[1] * (i / self.bezier_n)
            base_px_val = self.target_point[0] * (i / self.bezier_n)

            if i == 1:
                px_val = self.initial_point[0]
            else:
                px_val = base_px_val + self.np_random.uniform(-self.x_variation_amplitude, self.x_variation_amplitude)
                px_val = np.clip(px_val,
                                 min(self.initial_point[0], self.target_point[0]) - self.x_variation_amplitude,
                                 max(self.initial_point[0], self.target_point[0]) + self.x_variation_amplitude)

            px_val = np.clip(px_val, min_x_world_padded, max_x_world_padded)
            py_val = np.clip(py_val, min_y_world_padded, max_y_world_padded)
            control_points.append([px_val, py_val])

        control_points.append(self.target_point.tolist())
        return control_points

    def _generate_obstacles_outside_band(self, num_attempts: int = 1500) -> List[np.ndarray]:
        obstacles: List[np.ndarray] = []
        if not self.internal_reference_control_points:
            return obstacles

        path_samples = np.array([
            BezierCurveUtils.bezier_point(self.internal_reference_control_points, t)
            for t in np.linspace(0, 1, 200)
        ])

        world_area = (self.world_bounds_x[1] - self.world_bounds_x[0]) * \
                     (self.world_bounds_y[1] - self.world_bounds_y[0])
        _, _, path_length = BezierCurveUtils.create_arc_length_lookup_table(self.internal_reference_control_points)
        corridor_area = path_length * self.band_width
        available_area = max(0, world_area - corridor_area)
        avg_radius = (self.min_obs_radius + self.max_obs_radius) / 2.0
        avg_obs_area = np.pi * avg_radius**2
        num_target_obstacles = int((available_area * self.obstacle_density) / avg_obs_area) if avg_obs_area > 1e-6 else 0
        if self.verbose:
            print(f"Targeting {num_target_obstacles} obstacles.")

        for _ in range(num_attempts):
            if len(obstacles) >= num_target_obstacles * 2 and len(obstacles) > 10:
                break

            obs_x = self.np_random.uniform(*self.world_bounds_x)
            obs_y = self.np_random.uniform(*self.world_bounds_y)
            obs_radius = self.np_random.uniform(self.min_obs_radius, self.max_obs_radius)
            new_obs_center = np.array([obs_x, obs_y])

            min_dist_to_path = np.min(np.linalg.norm(path_samples - new_obs_center, axis=1))
            if min_dist_to_path < (self.band_width / 2.0 + obs_radius):
                continue

            is_colliding = any(
                np.linalg.norm(new_obs_center - existing_obs[:2]) < (obs_radius + existing_obs[2] + 0.1)
                for existing_obs in obstacles
            )
            if is_colliding:
                continue

            if np.linalg.norm(new_obs_center - self.initial_point) < (obs_radius + 10.0):
                continue
            if np.linalg.norm(new_obs_center - self.target_point) < (obs_radius + 10.0):
                continue

            obstacles.append(np.array([obs_x, obs_y, obs_radius], dtype=np.float32))

        if self.verbose:
            print(f"Generated {len(obstacles)} obstacles.")
        return obstacles

    def generate_environment(self) -> Dict[str, Any]:
        self._update_random_state()
        self.target_point = self._generate_random_target_point()
        self.internal_reference_control_points = self._generate_internal_reference_bezier_control_points()
        self.generated_obstacles = self._generate_obstacles_outside_band()

        return {
            "target_point": self.target_point.tolist(),
            "obstacles": [obs.tolist() for obs in self.generated_obstacles],
            "band_width": self.band_width,
            "obstacle_density": self.obstacle_density,
            "obstacle_count": len(self.generated_obstacles),
            "generation_seed": self._seed_counter
        }


# ==================== Main RL Environment ====================

GLOBAL_PERCEPTION_COMPONENT: Optional[OptimizedPerceptionModule] = None

def get_perception_component(model_path: str) -> OptimizedPerceptionModule:
    global GLOBAL_PERCEPTION_COMPONENT
    if GLOBAL_PERCEPTION_COMPONENT is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Perception model not found at path: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        GLOBAL_PERCEPTION_COMPONENT = OptimizedPerceptionModule(
            model_path, ROBOT_G, ROBOT_B, device, verbose=True
        )
    return GLOBAL_PERCEPTION_COMPONENT


class MyShipEnv(gym.Env):
    """A Gymnasium environment for USV navigation using a Bezier curve-based planner."""
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 perception_model_path: str,
                 env_generator: Optional[ObstacleEnvironmentGenerator] = None,
                 environment_reuse_probability: float = 0.0,
                 max_collisions_per_env: int = 5):
        super().__init__()
        self.env_generator = env_generator
        self.perception_component = get_perception_component(perception_model_path)
        self.lidar_scanner = LidarScanner(
            perception_module=self.perception_component,
            use_perception=True
        )
        self.environment_reuse_probability = environment_reuse_probability
        self.max_collisions_per_env = max_collisions_per_env
        self.last_environment_config: Optional[Dict[str, Any]] = None
        self.current_env_collision_count: int = 0
        
        # --- Physical Limits (Added missing attributes) ---
        self.v_max = V_MAX
        self.v_min = V_MIN
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.goal_threshold = THRESHOLD_GOAL_DIST
        self.bezier_n = BEZIER_N
        self.RL_planning_interval_delta_T = RL_PLANNING_INTERVAL_DELTA_T
        self.low_level_control_dt = LOW_LEVEL_CONTROL_DT
        
        self.ship_state: np.ndarray = self._initial_ship_state()
        self.current_episode_goal_world: np.ndarray = np.zeros(2, dtype=np.float32)
        self.obstacles: List[np.ndarray] = []
        self.current_step: int = 0
        self.initial_goal_dist: float = 0.0
        self.previous_goal_dist: float = 0.0
        self.bezier_control_points: Optional[List[List[float]]] = None
        self.previous_bezier_control_points: Optional[List[List[float]]] = None
        self.t_from_arc_length_interp = None
        self.arc_length_from_t_interp = None
        self.current_bezier_curve_total_length: float = 0.0
        self.virtual_point_current_t: float = 0.0
        self.virtual_point_current_speed: float = 0.0
        self.current_pointcloud_body: Optional[np.ndarray] = None
        self._define_spaces()

    def _define_spaces(self) -> None:
        self.lidar_sector_angles = self._create_lidar_sector_angles()
        num_sectors = len(self.lidar_sector_angles)
        obs_dim = 1 + 2 + num_sectors * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(BEZIER_N,), dtype=np.float32)

    def _create_lidar_sector_angles(self) -> List[Tuple[float, float]]:
        angles = []
        for i in range(6):
            start_angle = -np.pi/6 + i * (np.pi / 18)
            end_angle = -np.pi/6 + (i + 1) * (np.pi / 18)
            angles.append((start_angle, end_angle))
        for i in range(3):
            start_angle = np.pi/6 + i * (np.pi / 9)
            end_angle = np.pi/6 + (i + 1) * (np.pi / 9)
            angles.append((start_angle, end_angle))
        for i in range(3):
            start_angle = -np.pi/2 + i * (np.pi / 9)
            end_angle = -np.pi/2 + (i + 1) * (np.pi / 9)
            angles.append((start_angle, end_angle))
        angle_rear = np.arctan(8 / 5)
        angles.append((np.pi/2, np.pi/2 + angle_rear))
        angles.append((-np.pi/2 - angle_rear, -np.pi/2))
        return angles

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        should_generate_new_env = (
            self.last_environment_config is None or
            self.current_env_collision_count >= self.max_collisions_per_env
        )

        if should_generate_new_env:
            self.current_env_collision_count = 0
            if self.np_random.random() < self.environment_reuse_probability and self.last_environment_config is not None:
                pass # Reuse last config
            elif self.env_generator:
                self.last_environment_config = self.env_generator.generate_environment()
            else: # Fallback to a default environment if no generator and no last config
                self.last_environment_config = {
                    'target_point': [150.0, 190.0],
                    'obstacles': [[120, 100, 10], [180, 140, 8]]
                }

        self.current_episode_goal_world = np.array(self.last_environment_config['target_point'], dtype=np.float32)
        self.obstacles = [np.array(obs, dtype=np.float32) for obs in self.last_environment_config['obstacles']]
        
        self.ship_state = self._initial_ship_state()
        self.virtual_point_current_t = 0.0
        self.virtual_point_current_speed = 0.0
        self.current_step = 0
        self.bezier_control_points = None
        self.previous_bezier_control_points = None
        self.initial_goal_dist = np.linalg.norm(self.ship_state[:2] - self.current_episode_goal_world)
        if self.initial_goal_dist < 1e-6: self.initial_goal_dist = 1.0
        self.previous_goal_dist = self.initial_goal_dist
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        
        a_normalized = action[0]
        a_target = a_normalized * A_MAX
        v_current = self.virtual_point_current_speed
        v_next_unclipped = v_current + a_target * RL_PLANNING_INTERVAL_DELTA_T
        v_next_achievable = np.clip(v_next_unclipped, self.v_min, self.v_max)
        a_target_final = (v_next_achievable - v_current) / RL_PLANNING_INTERVAL_DELTA_T

        continue_previous_path = self._decide_and_plan_path(action, v_current, v_next_achievable)
        
        done, info = self._simulate_physics(a_target_final)
        
        reward = self._calculate_reward(self.ship_state, v_current, continue_previous_path)
        self.previous_goal_dist = np.linalg.norm(self.ship_state[:2] - self.current_episode_goal_world)

        if not done:
            dist_to_goal = np.linalg.norm(self.ship_state[:2] - self.current_episode_goal_world)
            if dist_to_goal < THRESHOLD_GOAL_DIST:
                done = True
                info["is_success"] = True
                reward += REWARD_GOAL_REACHED
                self.current_env_collision_count = self.max_collisions_per_env
        
        if done and info.get("collided", False):
            self.current_env_collision_count += 1
            reward += REWARD_COLLISION

        truncated = self.current_step >= MAX_EPISODE_STEPS
        
        return self._get_obs(), reward, done, truncated, info

    def _decide_and_plan_path(self, action: np.ndarray, v_current: float, v_next: float) -> bool:
        continue_previous_path = False
        if self.previous_bezier_control_points:
            dist_to_goal = np.linalg.norm(np.array(self.previous_bezier_control_points[-1]) - self.current_episode_goal_world)
            if dist_to_goal < self.v_max * RL_PLANNING_INTERVAL_DELTA_T and self.virtual_point_current_t < 1.0:
                continue_previous_path = True
                self.bezier_control_points = self.previous_bezier_control_points
        
        if not continue_previous_path:
            self._plan_new_path(action, v_current, v_next)

        self.previous_bezier_control_points = self.bezier_control_points.copy()
        return continue_previous_path

    def _plan_new_path(self, action: np.ndarray, v_current: float, v_next: float) -> None:
        base_length = 0.5
        px_actions_normalized = action[1:]
        average_speed = (v_current + v_next) / 2.0
        speed_scaling_factor = average_speed + 1e-6
        P0 = self.ship_state[:2]
        current_yaw = self.ship_state[4]
        cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)
        
        control_points: List[List[float]] = [P0.tolist()]
        cumulative_y = 0.0

        P1_rel_body_y = ALPHA_I[1] * RL_PLANNING_INTERVAL_DELTA_T * speed_scaling_factor
        cumulative_y += P1_rel_body_y
        P1_rel_body_x = 0.0
        P1_rel_world_x = P1_rel_body_x * cos_yaw - P1_rel_body_y * sin_yaw
        P1_rel_world_y = P1_rel_body_x * sin_yaw + P1_rel_body_y * cos_yaw
        control_points.append((P0 + np.array([P1_rel_world_x, P1_rel_world_y])).tolist())

        for i in range(2, BEZIER_N + 1):
            delta_y = ALPHA_I[i] * RL_PLANNING_INTERVAL_DELTA_T * speed_scaling_factor
            cumulative_y += delta_y + base_length
            py_val = cumulative_y
            px_action_normalized = px_actions_normalized[i - 2] if i - 2 < len(px_actions_normalized) else 0.0
            max_px_offset = 0.1 * py_val
            px_val = px_action_normalized * max_px_offset
            P_i_rel_world_x = px_val * cos_yaw - py_val * sin_yaw
            P_i_rel_world_y = px_val * sin_yaw + py_val * cos_yaw
            control_points.append((P0 + np.array([P_i_rel_world_x, P_i_rel_world_y])).tolist())

        self.bezier_control_points = control_points
        self.t_from_arc_length_interp, self.arc_length_from_t_interp, self.current_bezier_curve_total_length = \
            BezierCurveUtils.create_arc_length_lookup_table(self.bezier_control_points)
        self.virtual_point_current_t = 0.0

    def _simulate_physics(self, acceleration: float) -> Tuple[bool, Dict]:
        num_substeps = int(RL_PLANNING_INTERVAL_DELTA_T / LOW_LEVEL_CONTROL_DT) or 1
        
        for _ in range(num_substeps):
            v_start_substep = self.virtual_point_current_speed
            self.virtual_point_current_speed = np.clip(
                v_start_substep + acceleration * LOW_LEVEL_CONTROL_DT, self.v_min, self.v_max
            )
            delta_s = (v_start_substep + self.virtual_point_current_speed) / 2.0 * LOW_LEVEL_CONTROL_DT
            
            current_arc_len = self.arc_length_from_t_interp(self.virtual_point_current_t)
            target_arc_len = np.clip(current_arc_len + delta_s, 0, self.current_bezier_curve_total_length)
            
            self.virtual_point_current_t = float(self.t_from_arc_length_interp(target_arc_len))
            
            new_pos = BezierCurveUtils.bezier_point(self.bezier_control_points, self.virtual_point_current_t)
            tangent = BezierCurveUtils.bezier_derivative_point(self.bezier_control_points, self.virtual_point_current_t)
            
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-6:
                direction = tangent / tangent_norm
                new_vel = self.virtual_point_current_speed * direction
                new_yaw = np.arctan2(-new_vel[0], new_vel[1]) 
            else:
                new_vel = self.ship_state[2:4]
                new_yaw = self.ship_state[4]

            self.ship_state = np.array([new_pos[0], new_pos[1], new_vel[0], new_vel[1], new_yaw], dtype=np.float32)

            if self._check_collision():
                return True, {"collided": True}
                
        return False, {}

    def _get_obs(self) -> np.ndarray:
        ship_pos, ship_vel, ship_yaw = self.ship_state[:2], self.ship_state[2:4], self.ship_state[4]
        current_speed = np.linalg.norm(ship_vel)
        rel_goal_world = self.current_episode_goal_world - ship_pos
        cos_yaw, sin_yaw = np.cos(ship_yaw), np.sin(ship_yaw)
        g_body_x = rel_goal_world[0] * cos_yaw + rel_goal_world[1] * sin_yaw
        g_body_y = -rel_goal_world[0] * sin_yaw + rel_goal_world[1] * cos_yaw
        g_body = np.array([g_body_x, g_body_y], dtype=np.float32)
        
        self.current_pointcloud_body, distances = self.lidar_scanner.scan_environment(ship_pos, ship_yaw, self.obstacles)
        lidar_features = self.lidar_scanner.get_sector_features(self.current_pointcloud_body, distances, self.lidar_sector_angles)

        return np.concatenate([
            np.array([current_speed], dtype=np.float32),
            g_body,
            lidar_features
        ]).astype(np.float32)
    
    def _calculate_reward(self, current_ship_state: np.ndarray, v_current: float, continue_previous_path: bool) -> float:
        ship_pos, v_t = current_ship_state[:2], np.linalg.norm(current_ship_state[2:4])
        current_goal_dist = np.linalg.norm(ship_pos - self.current_episode_goal_world)
        
        if continue_previous_path:
            r_goal = 5 * EPSILON_1 * (self.previous_goal_dist - current_goal_dist) / self.initial_goal_dist
            r_path_approaching = 0.0
        else:
            r_goal = 0.8 * EPSILON_1 * (self.previous_goal_dist - current_goal_dist) / self.initial_goal_dist
            if self.bezier_control_points:
                P_N = np.array(self.bezier_control_points[-1])
                dist_PN_to_goal = np.linalg.norm(P_N - self.current_episode_goal_world)
                r_path_approaching = -EPSILON_2 * (dist_PN_to_goal - current_goal_dist) / self.initial_goal_dist
            else:
                r_path_approaching = -EPSILON_2 * 10.0

        reward = r_goal + r_path_approaching

        if self.bezier_control_points:
            safety_penalty = self._calculate_path_safety_penalty(continue_previous_path)
            curvature_penalty = self._calculate_curvature_penalty(continue_previous_path)
            if v_t < 1:
                curvature_penalty *= 0.5
            reward += safety_penalty + curvature_penalty
        else:
            reward += -EPSILON_3 * 20.0

        if v_t > 1:
            r_stable = -EPSILON_5 * np.abs(v_current - v_t)
        elif v_t - v_current > 0:
            r_stable = 5 * EPSILON_5 * (v_t - v_current)
        else:
            r_stable = 0.0
        reward += r_stable

        reward -= 0.05
        return reward

    def _calculate_path_safety_penalty(self, partial: bool) -> float:
        if self.current_pointcloud_body is None or self.current_pointcloud_body.shape[0] == 0:
            return 0.0
        
        current_t = self.virtual_point_current_t
        start_j = max(1, int(current_t * PATH_SAMPLE_POINTS_M)) if partial else 1
        num_points_to_check = PATH_SAMPLE_POINTS_M + 1 - start_j
        if num_points_to_check <= 0: return 0.0

        t_values = np.linspace(start_j / PATH_SAMPLE_POINTS_M, 1.0, num_points_to_check)
        path_points = np.array([BezierCurveUtils.bezier_point(self.bezier_control_points, t) for t in t_values])
        tangents = np.array([BezierCurveUtils.bezier_derivative_point(self.bezier_control_points, t) for t in t_values])
        
        valid_mask = np.linalg.norm(tangents, axis=1) > 1e-6
        path_yaws = np.zeros(len(t_values))
        path_yaws[valid_mask] = np.arctan2(-tangents[valid_mask, 0], tangents[valid_mask, 1])
        path_yaws[~valid_mask] = self.ship_state[4]
        
        ship_pos, ship_yaw = self.ship_state[:2], self.ship_state[4]
        pointcloud_world = self._body_to_world_coordinates_vectorized(self.current_pointcloud_body, ship_pos, ship_yaw)
        
        all_transformed_points = [
            self._world_to_body_coordinates_vectorized(pointcloud_world, p_path, p_yaw)
            for p_path, p_yaw in zip(path_points, path_yaws)
        ]
        
        if not all_transformed_points: return 0.0
        
        batch_points = np.vstack(all_transformed_points)
        batch_distances = self.perception_component.predict_distance_batch(batch_points)
        batch_distances = batch_distances.reshape(num_points_to_check, len(self.current_pointcloud_body))
        min_distances = np.min(batch_distances, axis=1)

        DANGER_ZONE = 5.0
        unsafe_mask = min_distances < DANGER_ZONE
        
        if np.any(unsafe_mask):
            penalty_factors = np.square(1 - min_distances[unsafe_mask] / DANGER_ZONE)
            return -EPSILON_3 * np.sum(penalty_factors)
        return 0.0

    def _calculate_curvature_penalty(self, partial: bool) -> float:
        if not self.bezier_control_points: return 0.0

        t_samples = np.linspace(0, 1, PATH_SAMPLE_POINTS_M + 1)
        tangents = np.array([BezierCurveUtils.bezier_derivative_point(self.bezier_control_points, t) for t in t_samples])
        
        valid_mask = np.linalg.norm(tangents, axis=1) > 1e-6
        if np.sum(valid_mask) < 2: return 0.0

        yaws_rad = np.zeros_like(t_samples)
        yaws_rad[valid_mask] = np.arctan2(-tangents[valid_mask, 0], tangents[valid_mask, 1])
        
        if not valid_mask[0]: yaws_rad[0] = self.ship_state[4]
        for i in range(1, len(yaws_rad)):
            if not valid_mask[i]: yaws_rad[i] = yaws_rad[i-1]

        yaw_diffs_rad = yaws_rad[1:] - yaws_rad[:-1]
        yaw_diffs_rad = np.arctan2(np.sin(yaw_diffs_rad), np.cos(yaw_diffs_rad))
        yaw_diffs_deg = np.rad2deg(yaw_diffs_rad)
        
        start_idx = int(self.virtual_point_current_t * PATH_SAMPLE_POINTS_M) if partial else 0
        
        total_penalty = 0.0
        if start_idx < len(yaw_diffs_deg):
            total_penalty = np.sum(np.square(yaw_diffs_deg[start_idx:]))
        
        return -EPSILON_4 * total_penalty

    def _initial_ship_state(self) -> np.ndarray:
        return np.array([100.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _check_collision(self) -> bool:
        ship_pos, ship_yaw = self.ship_state[:2], self.ship_state[4]
        _, distances = self.lidar_scanner.scan_environment(ship_pos, ship_yaw, self.obstacles)
        return np.min(distances) < COLLISION_THRESHOLD if distances.size > 0 else False
    
    def _body_to_world_coordinates_vectorized(self, points_body, ship_pos, ship_yaw):
        cos_y, sin_y = np.cos(ship_yaw), np.sin(ship_yaw)
        rot_matrix = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        return (rot_matrix @ points_body.T).T + ship_pos

    def _world_to_body_coordinates_vectorized(self, points_world, ship_pos, ship_yaw):
        rel_points = points_world - ship_pos
        cos_y, sin_y = np.cos(ship_yaw), np.sin(ship_yaw)
        rot_matrix_inv = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
        return (rot_matrix_inv @ rel_points.T).T

    def render(self, mode='human'):
        pass

    def visualize_episode_trajectory(self, trajectory_states: List[np.ndarray], save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor("#e0e0e0")

        if hasattr(self, 'obstacles') and self.obstacles:
            for obs in self.obstacles:
                ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color='#F35A5A', alpha=1))
        
        if hasattr(self, 'current_episode_goal_world') and self.current_episode_goal_world is not None:
            ax.add_patch(plt.Circle(self.current_episode_goal_world, THRESHOLD_GOAL_DIST, 
                                     facecolor='green', alpha=0.3, label='Goal Area'))
        
        if trajectory_states:
            positions = np.array([state[:2] for state in trajectory_states])
            ax.plot(positions[:, 0], positions[:, 1], 'g--', linewidth=2, label='Ship Trajectory')
            ax.plot(positions[0, 0], positions[0, 1], 'yo', markersize=8, label='Start')
            ax.plot(positions[-1, 0], positions[-1, 1], 'bo', markersize=8, label='End')
        
        ax.set_title("Episode Trajectory")
        ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)")
        ax.set_xlim(200), ax.set_ylim(200)
        ax.set_aspect('equal', adjustable='box'), ax.legend(), ax.grid(True, linestyle=':', color='white')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

# --- END OF FILE ship_rl_planner/environment.py ---