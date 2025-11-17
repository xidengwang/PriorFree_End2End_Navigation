# --- START OF FILE ship_rl_planner/utils/bezier.py ---
"""
A utility library for calculations related to Bezier curves.

This file provides a static class `BezierCurveUtils` that offers high-performance
functions for evaluating points, derivatives, and arc lengths of Bezier curves.
Core mathematical operations are JIT-compiled using Numba for speed.
"""

from typing import Tuple, Callable, List

import numpy as np
from numba import njit
from scipy.interpolate import interp1d

# ==================== Numba JIT-Compiled Core Functions ====================

@njit(cache=True)
def combinations(n: int, k: int) -> int:
    """High-performance, Numba-compatible combination function (nCk)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

@njit(cache=True)
def _bernstein_polynomial_njit(n: int, i: int, t: float) -> float:
    """Computes the value of a Bernstein polynomial."""
    if t < 0.0 or t > 1.0:
        t = max(0.0, min(1.0, t))
    comb = combinations(n, i)
    return comb * (t ** i) * ((1.0 - t) ** (n - i))

@njit(cache=True)
def bezier_point_njit(control_points: np.ndarray, t: float) -> np.ndarray:
    """Computes a point on a Bezier curve for a given parameter t."""
    n = len(control_points) - 1
    if n < 0:
        return np.zeros(2, dtype=np.float64)
        
    point = np.zeros(2, dtype=np.float64)
    for i in range(n + 1):
        bernstein_coeff = _bernstein_polynomial_njit(n, i, t)
        point += bernstein_coeff * control_points[i]
    return point

@njit(cache=True)
def bezier_derivative_point_njit(control_points: np.ndarray, t: float) -> np.ndarray:
    """Computes the first derivative (tangent vector) of a Bezier curve."""
    n = len(control_points) - 1
    if n < 1:
        return np.zeros(2, dtype=np.float64)
    
    point = np.zeros(2, dtype=np.float64)
    for i in range(n):
        bernstein_coeff = _bernstein_polynomial_njit(n - 1, i, t)
        point += bernstein_coeff * (control_points[i + 1] - control_points[i])
    return n * point

@njit(cache=True)
def bezier_second_derivative_point_njit(control_points: np.ndarray, t: float) -> np.ndarray:
    """Computes the second derivative of a Bezier curve."""
    n = len(control_points) - 1
    if n < 2:
        return np.zeros(2, dtype=np.float64)
    
    point = np.zeros(2, dtype=np.float64)
    for i in range(n - 1):
        bernstein_coeff = _bernstein_polynomial_njit(n - 2, i, t)
        term = control_points[i + 2] - 2 * control_points[i + 1] + control_points[i]
        point += bernstein_coeff * term
    return n * (n - 1) * point


# ==================== Main Utility Class ====================

class BezierCurveUtils:
    """A static utility class for Bezier curve operations."""

    @staticmethod
    def bezier_point(control_points: List[List[float]], t: float) -> np.ndarray:
        return bezier_point_njit(np.asarray(control_points, dtype=np.float64), t)

    @staticmethod
    def bezier_derivative_point(control_points: List[List[float]], t: float) -> np.ndarray:
        return bezier_derivative_point_njit(np.asarray(control_points, dtype=np.float64), t)

    @staticmethod
    def bezier_second_derivative_point(control_points: List[List[float]], t: float) -> np.ndarray:
        return bezier_second_derivative_point_njit(np.asarray(control_points, dtype=np.float64), t)

    @staticmethod
    def create_arc_length_lookup_table(
        control_points: List[List[float]], num_samples: int = 1000
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], float]:
        t_samples = np.linspace(0, 1, num_samples)
        arc_lengths = np.zeros(num_samples)
        
        control_points_arr = np.asarray(control_points, dtype=np.float64)
        n = len(control_points_arr) - 1
        
        if n >= 1:
            derivative_control_points = n * (control_points_arr[1:] - control_points_arr[:-1])
            i = np.arange(n)
            combs = np.array([combinations(n - 1, j) for j in i])
            
            t_matrix = t_samples[:, np.newaxis]
            bernstein_coeffs = combs * (t_matrix ** i) * ((1 - t_matrix) ** (n - 1 - i))
            
            all_derivatives = bernstein_coeffs @ derivative_control_points
            all_speeds = np.linalg.norm(all_derivatives, axis=1)
            
            segment_lengths = (all_speeds[:-1] + all_speeds[1:]) / 2.0 * (t_samples[1] - t_samples[0])
            arc_lengths[1:] = np.cumsum(segment_lengths)
        
        total_arc_length = arc_lengths[-1]

        unique_arc_lengths, unique_indices = np.unique(arc_lengths, return_index=True)
        unique_t_samples = t_samples[unique_indices]

        if len(unique_arc_lengths) < 2:
            s_from_t = lambda t_val: np.zeros_like(t_val) if isinstance(t_val, np.ndarray) else 0.0
            t_from_s = lambda s_val: np.zeros_like(s_val) if isinstance(s_val, np.ndarray) else 0.0
            return t_from_s, s_from_t, 0.0
        
        s_from_t = interp1d(
            t_samples, arc_lengths,
            kind='linear',
            bounds_error=False,
            fill_value=(arc_lengths[0], arc_lengths[-1])
        )
        t_from_s = interp1d(
            unique_arc_lengths, unique_t_samples,
            kind='linear',
            bounds_error=False,
            fill_value=(unique_t_samples[0], unique_t_samples[-1])
        )
        
        return t_from_s, s_from_t, total_arc_length

# --- END OF FILE ship_rl_planner/utils/bezier.py ---