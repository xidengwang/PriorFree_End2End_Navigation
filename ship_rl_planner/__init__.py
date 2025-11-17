# --- START OF FILE ship_rl_planner/__init__.py ---
"""
Ship RL Planner: A Core Library for USV Motion Planning Research.

This package contains the essential components for the reinforcement learning
environment, perception modules, and simulation utilities used in our paper.

By importing key classes to the top level, we allow for cleaner imports in
the experiment and training scripts. For example, you can use:

from ship_rl_planner import MyShipEnv, BezierCurveUtils
"""

# Import key classes from submodules to make them directly accessible
# from the 'ship_rl_planner' package.
from .perception import Perc_Net, OptimizedPerceptionModule
from .utils.bezier import BezierCurveUtils
from .utils.lidar import LidarScanner
from .environment import MyShipEnv, ObstacleEnvironmentGenerator
from .fixed_env_generator import fixed_env

# Define the public API of this package.
__all__ = [
    # From environment.py
    "MyShipEnv",
    "ObstacleEnvironmentGenerator",
    
    # From fixed_env_generator.py
    "fixed_env",
    
    # From perception.py
    "Perc_Net",
    "OptimizedPerceptionModule",
    
    # From utils/
    "BezierCurveUtils",
    "LidarScanner",
]
# --- END OF FILE ship_rl_planner/__init__.py ---