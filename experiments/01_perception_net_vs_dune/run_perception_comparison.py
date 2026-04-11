# --- START OF FILE experiments/01_perception_net_vs_dune/run_perception_comparison.py ---
"""
Compares the performance of our Perception Net against the DUNE baseline.
"""

import os
import time
import warnings
from typing import Tuple, Dict, Any, Optional, List, Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from tqdm import tqdm
import numba



class Perc_Net(nn.Module):
    """Defines the architecture for our Perception Network."""
    def __init__(self, G: np.ndarray, b: np.ndarray, input_size: int = 2, hidden_size: int = 32):
        super().__init__()
        self.output_size = G.shape[0]
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, self.output_size), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ObsPointNet(nn.Module):
    """Defines the architecture for the DUNE baseline model."""
    def __init__(self, input_dim: int = 2, output_dim: int = 4):
        super().__init__()
        hidden_dim = 32
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MLP(x)

@numba.jit(nopython=True, cache=True)
def _point_to_segment_distance_numba(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    """Numba-optimized function to find the shortest distance from a point to a line segment."""
    ab = b - a
    ap = p - a
    dot_ab_ab = np.dot(ab, ab)
    if dot_ab_ab < 1e-9:
        return np.sqrt(np.sum(ap**2)), a
    
    t = np.dot(ap, ab) / dot_ab_ab
    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    
    closest_point = a + t * ab
    dist_vec = p - closest_point
    return np.sqrt(np.sum(dist_vec**2)), closest_point

@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_distance_loop(points: np.ndarray, ordered_vertices: np.ndarray, G: np.ndarray, b_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-optimized loop to calculate distances for a batch of points to a polygon."""
    num_points = len(points)
    num_edges = len(ordered_vertices)
    distances = np.zeros(num_points, dtype=np.float64)
    closest_points_on_poly = np.zeros_like(points, dtype=np.float64)

    for i in numba.prange(num_points):
        point = points[i]
        is_inside = True
        for k in range(G.shape[0]):
            if np.dot(G[k], point) > b_flat[k]:
                is_inside = False
                break
        if is_inside:
            distances[i] = 0.0
            closest_points_on_poly[i] = point
            continue
        
        min_dist_for_point = np.inf
        closest_point_on_edge = point
        for j in range(num_edges):
            v1 = ordered_vertices[j]
            v2 = ordered_vertices[(j + 1) % num_edges]
            dist, current_closest = _point_to_segment_distance_numba(point, v1, v2)
            if dist < min_dist_for_point:
                min_dist_for_point = dist
                closest_point_on_edge = current_closest
        distances[i] = min_dist_for_point
        closest_points_on_poly[i] = closest_point_on_edge
    return distances, closest_points_on_poly

def calculate_ground_truth_distances(points: np.ndarray, G: np.ndarray, b: np.ndarray) -> Tuple[Optional[np.ndarray]]:
    """Calculates the exact geometric distance from points to the polygon Gx <= b."""
    x_min, x_max = -0.8, 3.8
    y_min, y_max = -0.8, 0.8
    ordered_vertices = np.array([
        [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
    ], dtype=np.float64)

    try:
        distances, _ = _numba_distance_loop(points, ordered_vertices, G, b.flatten())
        return distances
    except Exception as e:
        print(f"Error during Numba distance calculation: {e}")
        return None


def Perc_Net_distest(k: int, G: np.ndarray, b: np.ndarray, model_path: str, low: float, high: float) -> Optional[np.ndarray]:
    """Evaluates the Perception Net's accuracy and returns the absolute errors."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Perc_Net(G=G, b=b)
        
        original_state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {f"model.{key}": value for key, value in original_state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(device).eval()

        points = np.random.uniform(low, high, size=(k, 2)).astype(np.float64)
        gt_distances = calculate_ground_truth_distances(points, G, b)
        if gt_distances is None: return None

        input_tensor = torch.from_numpy(points).float().to(device)
        with torch.no_grad():
            predicted_mu = model(input_tensor)
        
        G_torch = torch.from_numpy(G).float().to(device)
        b_torch = torch.from_numpy(b).float().to(device)
        mu_col, p_col = predicted_mu.unsqueeze(-1), input_tensor.unsqueeze(-1)
        quad_form = mu_col.transpose(-2, -1) @ (G_torch @ (G_torch.T @ mu_col))
        linear_term = mu_col.transpose(-2, -1) @ (torch.bmm(G_torch.expand(k, -1, -1), p_col) - b_torch)
        
        nn_dist_sq = (-0.25 * quad_form + linear_term).squeeze().cpu().numpy()
        nn_dist = np.sqrt(np.maximum(0, nn_dist_sq))
        
        return np.abs(nn_dist - gt_distances)
    except Exception as e:
        print(f"Error evaluating Perc_Net accuracy: {e}")
        return None

def DUNE_distest(k: int, G: np.ndarray, b: np.ndarray, model_path: str, low: float, high: float) -> Optional[np.ndarray]:
    """Evaluates the DUNE model's accuracy and returns the absolute errors."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ObsPointNet(output_dim=G.shape[0])
        
        original_state_dict = torch.load(model_path, map_location=device, weights_only=True)
        new_state_dict = {
            key if key.startswith("MLP.") else f"MLP.{key}": value
            for key, value in original_state_dict.items()
        }
        model.load_state_dict(new_state_dict)
        model.to(device).eval()

        points = np.random.uniform(low, high, size=(k, 2)).astype(np.float64)
        gt_distances = calculate_ground_truth_distances(points, G, b)
        if gt_distances is None: return None

        input_tensor = torch.from_numpy(points).float().to(device)
        with torch.no_grad():
            mu_nn = model(input_tensor)

        G_torch = torch.from_numpy(G).float().to(device)
        b_torch = torch.from_numpy(b).float().to(device)
        mu_col, p_col = mu_nn.unsqueeze(-1), input_tensor.unsqueeze(-1)
        
        temp = torch.bmm(G_torch.expand(k, -1, -1), p_col) - b_torch
        nn_dist = torch.bmm(mu_col.transpose(-2, -1), temp).squeeze().cpu().numpy()
        nn_dist = np.maximum(0, nn_dist)

        return np.abs(nn_dist - gt_distances)
    except Exception as e:
        print(f"Error evaluating DUNE accuracy: {e}")
        return None

def benchmark_model_speed(model_class: type, model_path: str, G: np.ndarray, b: np.ndarray, 
                          sample_sizes: List[int], is_dune: bool = False) -> List[float]:
    """Measures the inference time for a given model over various sample sizes."""
    timings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBenchmarking {model_class.__name__} on {device}...")

    # Load model once
    if is_dune:
        model = model_class(output_dim=G.shape[0])
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        new_state_dict = {key if key.startswith("MLP.") else f"MLP.{key}": v for key, v in state_dict.items()}
    else:
        model = model_class(G=G, b=b)
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {f"model.{key}": v for key, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    
    for k in tqdm(sample_sizes, desc=f"Timing {model_class.__name__}"):
        points = np.random.uniform(-100, 100, size=(k, 2)).astype(np.float32)
        input_tensor = torch.from_numpy(points).to(device)

        # GPU warm-up
        for _ in range(2):
            _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        # Actual timing
        start_time = time.perf_counter()
        _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        timings.append(end_time - start_time)
        
    return timings

def plot_accuracy_panel(ax: plt.Axes, data: pd.DataFrame, quantile: float = 0.95) -> None:
    """Plots the accuracy comparison (left panel) on the given axes."""
    lower_q, upper_q = (1 - quantile) / 2, 1 - (1 - quantile) / 2
    summary = data.groupby(['scale', 'method'])['absolute_error'].agg(
        mean='mean', max='max', 
        q_lower=lambda x: x.quantile(lower_q), 
        q_upper=lambda x: x.quantile(upper_q)
    ).reset_index()

    palette = {"Perception Net (our)": "C1", "DUNE": "C0"}
    sns.lineplot(data=summary, x='scale', y='mean', hue='method', palette=palette, ax=ax, linewidth=3, legend=False)
    sns.lineplot(data=summary, x='scale', y='max', hue='method', palette=palette, ax=ax, linewidth=2, linestyle='--', legend=False)

    for method, color in palette.items():
        method_data = summary[summary['method'] == method]
        ax.fill_between(method_data['scale'], method_data['q_lower'], method_data['q_upper'], color=color, alpha=0.15)
        
    ax.set_yscale('log')
    ax.set_xlabel("Data Range Scale", fontsize=16)
    ax.set_ylabel("Absolute Error", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    
    mean_line = Line2D([0], [0], color='gray', lw=3, label='Mean Error')
    max_line = Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Maximum Error')
    quantile_patch = mpatches.Patch(color='gray', alpha=0.3, label=f'{int(quantile*100)}% Data Range')
    ax.legend(handles=[mean_line, max_line, quantile_patch], loc='lower right', fontsize=12)

def plot_speed_panel(ax: plt.Axes, speed_data: pd.DataFrame) -> Dict[str, str]:
    """Plots the speed comparison (right panel) using dynamically generated data."""
    color_map = {"Perception Net (our)": "C1", "DUNE": "C0"}
    
    sns.lineplot(data=speed_data, x='Samples (k)', y='Total Time (s)', hue='method', palette=color_map,
                 linewidth=3, marker='o', markersize=8, ax=ax, legend=False)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of Samples", fontsize=16)
    ax.set_ylabel("Total Inference Time (s)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    
    return color_map

def main():
    """Main function to run the comparison experiment."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ACKER_G = np.array([[0., -4.6], [1.6, 0.], [0., 4.6], [-1.6, 0.]], dtype=np.float64)
    ACKER_B = np.array([[3.68], [6.08], [3.68], [1.28]], dtype=np.float64)
    
    models_to_test = {
        "Perception Net (our)": {
            "path": os.path.join(SCRIPT_DIR, "../../pretrained_models/comparison_models/acker_vehicle/perception_net_acker.pth"),
            "eval_func": Perc_Net_distest,
            "model_class": Perc_Net,
            "is_dune": False,
        },
        "DUNE": {
            "path": os.path.join(SCRIPT_DIR, "../../pretrained_models/comparison_models/acker_vehicle/dune_acker.pth"),
            "eval_func": DUNE_distest,
            "model_class": ObsPointNet,
            "is_dune": True,
        }
    }
    
    # --- Part A: Data Generation for Accuracy Plot ---
    print("--- Part A: Starting Data Generation for Accuracy Comparison ---")
    K_SAMPLES_ACCURACY = 100_000
    RANGE_SCALES = np.linspace(5, 1000, 100)
    
    all_errors = []
    for scale in tqdm(RANGE_SCALES, desc="Accuracy | Processing Scales"):
        low, high = -scale, scale
        for method_name, config in models_to_test.items():
            if not os.path.exists(config["path"]):
                print(f"Warning: Model for '{method_name}' not found at {config['path']}. Skipping.")
                continue
                
            errors = config["eval_func"](k=K_SAMPLES_ACCURACY, G=ACKER_G, b=ACKER_B, 
                                        model_path=config["path"], low=low, high=high)
            if errors is not None:
                df = pd.DataFrame({'scale': scale, 'method': method_name, 'absolute_error': errors})
                all_errors.append(df)
    
    if not all_errors:
        print("\nFATAL ERROR: No accuracy data was generated. Exiting.")
        return
        
    df_all_errors = pd.concat(all_errors, ignore_index=True)
    print("--- Accuracy Data Generation Complete ---")

    # --- Part B: Data Generation for Speed Plot ---
    print("\n--- Part B: Starting Data Generation for Speed Comparison ---")
    SAMPLE_SIZES_SPEED = [5000, 10000, 50000, 100000, 500000, 1000000]
    speed_results = []
    for method_name, config in models_to_test.items():
        if not os.path.exists(config["path"]):
            continue
        timings = benchmark_model_speed(
            model_class=config["model_class"],
            model_path=config["path"],
            G=ACKER_G, b=ACKER_B,
            sample_sizes=SAMPLE_SIZES_SPEED,
            is_dune=config["is_dune"]
        )
        for k, t in zip(SAMPLE_SIZES_SPEED, timings):
            speed_results.append({"method": method_name, "Samples (k)": k, "Total Time (s)": t})
            
    if not speed_results:
        print("\nFATAL ERROR: No speed data was generated. Exiting.")
        return

    df_speed = pd.DataFrame(speed_results)
    print("--- Speed Data Generation Complete ---")

    # --- Part C: Plotting ---
    print("\n--- Part C: Generating Final Comparison Plot ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_accuracy_panel(ax=ax1, data=df_all_errors)
    color_map = plot_speed_panel(ax=ax2, speed_data=df_speed)
    
    legend_handles = [mpatches.Patch(color=color, label=method) for method, color in color_map.items()]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=len(color_map), fontsize=16, frameon=False)
               
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, "perception_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n--- Comparison plot successfully saved to: {output_path} ---")
    plt.show()

if __name__ == "__main__":
    main()
# --- END OF FILE experiments/01_perception_net_vs_dune/run_perception_comparison.py ---