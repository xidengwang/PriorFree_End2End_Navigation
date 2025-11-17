# --- START OF FILE ship_rl_planner/fixed_env_generator.py ---
"""
Defines a deterministic environment generator for reproducible experiments.

This file contains the `fixed_env` class, which inherits from
`ObstacleEnvironmentGenerator` but overrides the generation method to ensure
that the same environment (target, obstacles) is produced every time for a
given seed. This is essential for fair comparisons between different planning
algorithms.
"""

from typing import Dict, Any

from .environment import ObstacleEnvironmentGenerator


class fixed_env(ObstacleEnvironmentGenerator):
    """
    A deterministic version of the ObstacleEnvironmentGenerator.

    This class ensures that for a given initial `random_seed`, every call to
    `generate_environment()` will produce the identical environment layout.
    It achieves this by preventing the random state from being updated between
    calls, which is the behavior of the parent class.
    """

    def generate_environment(self) -> Dict[str, Any]:
        """
        Generates a fixed environment based on the initial seed.

        Unlike the parent class, this method does NOT update the random state,
        thus guaranteeing a deterministic output for a fixed seed.

        Returns:
            A dictionary containing the fixed target point and obstacle layout.
        """
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

# --- END OF FILE ship_rl_planner/fixed_env_generator.py ---