# --- START OF FILE ship_rl_planner/perception.py ---
"""
Perception components for the USV motion planner.

This file defines the neural network architecture (`Perc_Net`) and a high-level
handler (`OptimizedPerceptionModule`) for efficient, batch-processed distance
estimation from sensor points to the USV's geometric boundary.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class PointDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for handling point cloud data during model training."""

    def __init__(self, data_input: np.ndarray, data_lambda: np.ndarray, data_d: np.ndarray):
        self.data_input = data_input
        self.data_lambda = data_lambda
        self.data_d = data_d

    def __len__(self) -> int:
        return len(self.data_input)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.data_input[idx]).float(),
            torch.from_numpy(self.data_lambda[idx]).float(),
            torch.from_numpy(self.data_d[idx]).float(),
        )


class Perc_Net(nn.Module):
    """
    A perception neural network that predicts lambda values for distance calculation.
    """

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
        self.G_torch: Optional[torch.Tensor] = None
        self.b_torch: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class OptimizedPerceptionModule:
    """
    An integrated perception component that loads a pre-trained network and
    performs batch predictions of obstacle distances.
    """

    def __init__(self,
                 model_path: str,
                 G: np.ndarray,
                 b: np.ndarray,
                 device: Optional[torch.device] = None,
                 batch_size: int = 512,
                 verbose: bool = False):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        self.perc_net = Perc_Net(G=G, b=b)
        self._load_model_state(model_path, verbose)
        
        self.perc_net.G_torch = torch.from_numpy(G).float().to(self.device)
        self.perc_net.b_torch = torch.from_numpy(b).float().to(self.device)
        self.perc_net.to(self.device)
        self.perc_net.eval()
        
        self._prepare_inference_buffers()

    def _load_model_state(self, model_path: str, verbose: bool) -> None:
        try:
            if os.path.exists(model_path):
                if verbose:
                    print(f"Loading perception model from: {model_path} onto {self.device}")
                
                original_state_dict = torch.load(model_path, map_location='cpu')
                
                new_state_dict = {}
                for key, value in original_state_dict.items():
                    if not key.startswith("model."):
                        new_key = "model." + key
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                self.perc_net.load_state_dict(new_state_dict)

                if verbose:
                    print("Perception network loaded successfully.")
            else:
                if verbose:
                    print(f"Warning: Model file not found at {model_path}. Using an untrained network.")
        except Exception as e:
            if verbose:
                print(f"Error loading perception network: {e}. Using an untrained network.")

    def _prepare_inference_buffers(self) -> None:
        max_points = 4000
        self.input_buffer = torch.zeros((max_points, 2), dtype=torch.float32, device=self.device)

    def predict_distance_batch(self, obstacle_coords_body: np.ndarray) -> np.ndarray:
        n_points = obstacle_coords_body.shape[0]
        if n_points == 0:
            return np.array([], dtype=np.float32)

        if n_points <= self.input_buffer.size(0):
            input_tensor = self.input_buffer[:n_points]
            source_tensor = torch.from_numpy(obstacle_coords_body)
            input_tensor.copy_(source_tensor, non_blocking=True)
        else:
            input_tensor = torch.from_numpy(obstacle_coords_body).to(self.device, dtype=torch.float32)

        all_distances = []
        with torch.no_grad():
            for i in range(0, n_points, self.batch_size):
                end_idx = min(i + self.batch_size, n_points)
                batch_input = input_tensor[i:end_idx]
                
                predicted_lambda = self.perc_net(batch_input)
                distances = self._calculate_distances_vectorized(batch_input, predicted_lambda)
                all_distances.append(distances)

        final_distances = torch.cat(all_distances, dim=0)
        return final_distances.cpu().numpy()

    def _calculate_distances_vectorized(self, 
                                        obstacle_coords: torch.Tensor, 
                                        predicted_lambda: torch.Tensor) -> torch.Tensor:
        G_T_lambda = torch.einsum('ij,bj->bi', self.perc_net.G_torch.T, predicted_lambda)
        quad_form = torch.einsum('bi,ij,bj->b', predicted_lambda, self.perc_net.G_torch, G_T_lambda)
        
        G_obstacle = torch.einsum('ij,bj->bi', self.perc_net.G_torch, obstacle_coords)
        G_obstacle_minus_b = G_obstacle - self.perc_net.b_torch.squeeze()
        linear_term = torch.einsum('bi,bi->b', predicted_lambda, G_obstacle_minus_b)
        
        distances_squared = -0.25 * quad_form + linear_term
        
        distances_squared_positive = torch.relu(distances_squared)
        distances = torch.where(
            distances_squared_positive > 1e-8,
            torch.sqrt(distances_squared_positive),
            torch.zeros_like(distances_squared_positive)
        )
        
        return distances

# --- END OF FILE ship_rl_planner/perception.py ---