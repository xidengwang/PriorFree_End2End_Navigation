# --- START OF FILE ship_rl_planner/utils/lidar.py ---
"""
Defines a high-performance 2D LiDAR scanner simulation component.

This file contains the LidarScanner class, which is responsible for simulating
point cloud data by calculating intersections of laser beams with obstacles and
world boundaries in a 2D environment.
"""

import numpy as np
from typing import List, Tuple, Optional

# Forward declaration for type hinting
if False:
    from ..perception import OptimizedPerceptionModule


class LidarScanner:
    """A high-performance simulator for a 2D LiDAR sensor."""

    def __init__(self,
                 angular_resolution: float = 0.1,
                 max_range: float = 200.0,
                 world_bounds_x: Tuple[float, float] = (0.0, 200.0),
                 world_bounds_y: Tuple[float, float] = (0.0, 200.0),
                 perception_module: Optional['OptimizedPerceptionModule'] = None,
                 use_perception: bool = True):
        self.angular_resolution = angular_resolution
        self.max_range = max_range
        self.perception_module = perception_module
        self.use_perception = use_perception

        self.num_rays = int(360.0 / self.angular_resolution)
        self.scan_angles_nav = np.linspace(-np.pi, np.pi, self.num_rays, endpoint=False)
        
        self.boundaries = {
            'left': world_bounds_x[0], 'right': world_bounds_x[1], 
            'bottom': world_bounds_y[0], 'top': world_bounds_y[1]
        }
        
        self.pointcloud_body_buffer = np.zeros((self.num_rays, 2), dtype=np.float32)
        self.distances_buffer = np.zeros(self.num_rays, dtype=np.float32)
        self.pointcloud_world_buffer = np.zeros((self.num_rays, 2), dtype=np.float32)

    def _get_boundary_intersections(self, ship_pos: np.ndarray, ray_angles_math: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ship_x, ship_y = ship_pos
        ray_dx, ray_dy = np.cos(ray_angles_math), np.sin(ray_angles_math)
        intersections = np.zeros((self.num_rays, 2), dtype=np.float32)
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)

        for b_name, b_val in self.boundaries.items():
            if b_name in ['left', 'right']:
                valid_rays = np.abs(ray_dx) > 1e-10
                if not np.any(valid_rays): continue
                t = (b_val - ship_x) / ray_dx[valid_rays]
                forward = t > 1e-6
                if not np.any(forward): continue
                valid_t, valid_indices = t[forward], np.where(valid_rays)[0][forward]
                y_int = ship_y + valid_t * ray_dy[valid_indices]
                in_bounds = (y_int >= self.boundaries['bottom']) & (y_int <= self.boundaries['top'])
                if not np.any(in_bounds): continue
                final_indices, final_t, final_y = valid_indices[in_bounds], valid_t[in_bounds], y_int[in_bounds]
                update_mask = final_t < distances[final_indices]
                if not np.any(update_mask): continue
                update_indices, update_t, update_y = final_indices[update_mask], final_t[update_mask], final_y[update_mask]
                distances[update_indices] = update_t
                intersections[update_indices] = np.column_stack((np.full_like(update_y, b_val), update_y))
            else: # top, bottom
                valid_rays = np.abs(ray_dy) > 1e-10
                if not np.any(valid_rays): continue
                t = (b_val - ship_y) / ray_dy[valid_rays]
                forward = t > 1e-6
                if not np.any(forward): continue
                valid_t, valid_indices = t[forward], np.where(valid_rays)[0][forward]
                x_int = ship_x + valid_t * ray_dx[valid_indices]
                in_bounds = (x_int >= self.boundaries['left']) & (x_int <= self.boundaries['right'])
                if not np.any(in_bounds): continue
                final_indices, final_t, final_x = valid_indices[in_bounds], valid_t[in_bounds], x_int[in_bounds]
                update_mask = final_t < distances[final_indices]
                if not np.any(update_mask): continue
                update_indices, update_t, update_x = final_indices[update_mask], final_t[update_mask], final_x[update_mask]
                distances[update_indices] = update_t
                intersections[update_indices] = np.column_stack((update_x, np.full_like(update_x, b_val)))

        no_hit_mask = distances >= self.max_range
        if np.any(no_hit_mask):
            intersections[no_hit_mask, 0] = ship_x + self.max_range * ray_dx[no_hit_mask]
            intersections[no_hit_mask, 1] = ship_y + self.max_range * ray_dy[no_hit_mask]
            
        return intersections, distances

    def _get_obstacle_intersections(self, ship_pos: np.ndarray, ray_angles_math: np.ndarray, obstacles: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not obstacles:
            return np.full((self.num_rays, 2), np.inf, dtype=np.float32), np.full(self.num_rays, np.inf, dtype=np.float32)

        ship_x, ship_y = ship_pos
        ray_dx, ray_dy = np.cos(ray_angles_math), np.sin(ray_angles_math)
        obs_arr = np.array(obstacles)
        obs_x, obs_y, obs_r = obs_arr[:, 0], obs_arr[:, 1], obs_arr[:, 2]

        ray_dx_exp, ray_dy_exp = ray_dx[:, np.newaxis], ray_dy[:, np.newaxis]
        oc_x, oc_y = obs_x - ship_x, obs_y - ship_y

        center_dist_sq = oc_x**2 + oc_y**2
        ray_to_center_dot = ray_dx_exp * oc_x + ray_dy_exp * oc_y
        
        discriminant = ray_to_center_dot**2 - (center_dist_sq - obs_r**2)
        
        t = np.full_like(discriminant, np.inf)
        has_intersection = discriminant >= 0
        
        sqrt_discriminant = np.sqrt(discriminant[has_intersection])
        t1 = ray_to_center_dot[has_intersection] - sqrt_discriminant
        t2 = ray_to_center_dot[has_intersection] + sqrt_discriminant
        
        t_sol = np.where(t1 > 1e-6, t1, t2)
        t[has_intersection] = np.where(t_sol > 1e-6, t_sol, np.inf)
        
        min_dist_indices = np.argmin(t, axis=1)
        min_distances = t[np.arange(self.num_rays), min_dist_indices]
        
        intersections = np.zeros((self.num_rays, 2), dtype=np.float32)
        valid_mask = min_distances < np.inf
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            intersections[valid_indices, 0] = ship_x + min_distances[valid_mask] * ray_dx[valid_mask]
            intersections[valid_indices, 1] = ship_y + min_distances[valid_mask] * ray_dy[valid_mask]
        
        return intersections, min_distances

    def scan_environment(self, ship_position: np.ndarray, ship_yaw: float, obstacles: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        world_ray_angles_nav = self.scan_angles_nav + ship_yaw
        world_ray_angles_math = -world_ray_angles_nav + np.pi / 2
        
        boundary_intersections, boundary_distances = self._get_boundary_intersections(ship_position, world_ray_angles_math)
        obstacle_intersections, obstacle_distances = self._get_obstacle_intersections(ship_position, world_ray_angles_math, obstacles)
        
        use_obstacle = obstacle_distances < boundary_distances
        final_intersections = np.where(use_obstacle[:, np.newaxis], obstacle_intersections, boundary_intersections)
        final_distances = np.where(use_obstacle, obstacle_distances, boundary_distances)
        
        exceed_max_range = final_distances > self.max_range
        if np.any(exceed_max_range):
            final_distances[exceed_max_range] = self.max_range
            ray_dx = np.cos(world_ray_angles_math[exceed_max_range])
            ray_dy = np.sin(world_ray_angles_math[exceed_max_range])
            final_intersections[exceed_max_range, 0] = ship_position[0] + self.max_range * ray_dx
            final_intersections[exceed_max_range, 1] = ship_position[1] + self.max_range * ray_dy

        pointcloud_body = self._world_to_body_coordinates_vectorized(final_intersections, ship_position, ship_yaw)
        
        if self.use_perception and self.perception_module:
            perception_distances = self.perception_module.predict_distance_batch(pointcloud_body)
            return pointcloud_body.copy(), perception_distances
        else:
            return pointcloud_body.copy(), final_distances.copy()

    def get_sector_features(self, pointcloud_body: np.ndarray, distances: np.ndarray, sector_angles: List[Tuple[float, float]]) -> np.ndarray:
        point_angles_nav = np.arctan2(-pointcloud_body[:, 0], pointcloud_body[:, 1])
        
        sector_features = []
        for start_angle, end_angle in sector_angles:
            start_norm = self._normalize_angle(start_angle)
            end_norm = self._normalize_angle(end_angle)
            
            if start_norm <= end_norm:
                mask = (point_angles_nav >= start_norm) & (point_angles_nav < end_norm)
            else:
                mask = (point_angles_nav >= start_norm) | (point_angles_nav < end_norm)
            
            if np.any(mask):
                sector_distances = distances[mask]
                sector_point_angles = point_angles_nav[mask]
                min_idx = np.argmin(sector_distances)
                sector_features.extend([sector_distances[min_idx], sector_point_angles[min_idx]])
            else:
                sector_features.extend([self.max_range, 0.0])
                
        return np.array(sector_features, dtype=np.float32)
    
    def _normalize_angle(self, angle: float) -> float:
        while angle >= np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def _world_to_body_coordinates_vectorized(self, points_world: np.ndarray, ship_pos: np.ndarray, ship_yaw: float) -> np.ndarray:
        rel_points = points_world - ship_pos
        cos_yaw, sin_yaw = np.cos(ship_yaw), np.sin(ship_yaw)
        points_body = np.empty_like(rel_points)
        points_body[:, 0] = rel_points[:, 0] * cos_yaw + rel_points[:, 1] * sin_yaw
        points_body[:, 1] = -rel_points[:, 0] * sin_yaw + rel_points[:, 1] * cos_yaw
        return points_body
    
    def _body_to_world_coordinates_vectorized(self, points_world: np.ndarray, ship_pos: np.ndarray, ship_yaw: float) -> np.ndarray:
        rel_points = points_world - ship_pos
        cos_yaw, sin_yaw = np.cos(ship_yaw), np.sin(ship_yaw)
        points_body = np.empty_like(rel_points)
        points_body[:, 0] = rel_points[:, 0] * cos_yaw - rel_points[:, 1] * sin_yaw
        points_body[:, 1] = rel_points[:, 0] * sin_yaw + rel_points[:, 1] * cos_yaw
        return points_body

# --- END OF FILE ship_rl_planner/utils/lidar.py ---