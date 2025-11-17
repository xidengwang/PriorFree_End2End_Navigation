# --- START OF FILE training/train_sac_agent.py ---
"""
Main training script for the USV motion planner's SAC agent.

This script orchestrates the training process, including:
- Setting up the custom Gymnasium environment.
- Implementing a curriculum learning strategy with increasing difficulty.
- Configuring and launching the Stable Baselines3 SAC algorithm.
- Using a callback for displaying progress and periodic model saving.
"""

import os
import time
import multiprocessing
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

# Import core components from our custom library
from ship_rl_planner import MyShipEnv, ObstacleEnvironmentGenerator

# ==================== Training Configuration ====================

# --- Project Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, "logs/sac_ship_tensorboard/")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "training/checkpoints/")
FINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_models/rl_agent/sac_ship_planner_final.zip")
PERCEPTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_models/rl_agent/perception_net_ship.pth")

# --- SAC Hyperparameters (Consistent with original MySAC_train.py) ---
SAC_POLICY_KWARGS = dict(net_arch=[256, 256])
SAC_LEARNING_RATE = 3e-4
SAC_BUFFER_SIZE = 1_000_000
SAC_LEARNING_STARTS = 50_000
SAC_BATCH_SIZE = 2048
SAC_TAU = 0.005
SAC_GAMMA = 0.99
SAC_TRAIN_FREQ = (1, "step")
SAC_GRADIENT_STEPS = 1

# --- Curriculum & Environment ---
TOTAL_TIMESTEPS = 1_500_000
NUM_ENVS = 20
SAVE_INTERVAL = 100_000

# ==================== Custom Callback for Progress ====================

class ProgressCallback(BaseCallback):
    """
    A custom callback to display a progress bar and save model checkpoints periodically.
    Visualization is disabled as per user request.
    """
    def __init__(self, save_interval: int, model_dir: str, level_idx: int, total_level_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.model_dir = model_dir
        self.level_idx = level_idx
        self.total_level_steps = total_level_steps
        self.last_save_step = 0
        self.pbar: Optional[tqdm] = None

    def _on_training_start(self) -> None:
        """Called at the start of training for a level."""
        self.pbar = tqdm(total=self.total_level_steps, desc=f"Level {self.level_idx + 1}", unit="steps")
        self.last_save_step = self.num_timesteps

    def _on_step(self) -> bool:
        """Called at each step."""
        self.pbar.update(self.training_env.num_envs)
        
        if self.num_timesteps < self.model.learning_starts:
            status = f"Collecting ({self.num_timesteps}/{self.model.learning_starts})"
        else:
            status = "Learning"
        self.pbar.set_description(f"Level {self.level_idx + 1} - {status}")

        if (self.num_timesteps - self.last_save_step) >= self.save_interval:
            self.last_save_step = self.num_timesteps
            model_path = os.path.join(self.model_dir, f"model_{self.num_timesteps}_steps.zip")
            self.model.save(model_path)
            self.pbar.write(f"\n[Callback] Checkpoint saved to: {model_path}")
            
        return True

    def _on_training_end(self) -> None:
        """Called at the end of training for a level."""
        self.pbar.close()

# ==================== Main Training Logic ====================

def main():
    """Main function to run the curriculum training."""
    
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # --- Curriculum Learning Stages (Identical to original MySAC_train.py) ---
    curriculum_levels = [
        {
            'bezier_n': 3, 'band_width': 10.0, 'obstacle_density': 0.06,
            'x_variation_amplitude': 10.0, 'environment_reuse_probability': 0.7
        },
        {
            'bezier_n': 4, 'band_width': 8.0, 'obstacle_density': 0.12,
            'x_variation_amplitude': 20.0, 'environment_reuse_probability': 0.4
        },
        {
            'bezier_n': 5, 'band_width': 6.0, 'obstacle_density': 0.24,
            'x_variation_amplitude': 30.0, 'environment_reuse_probability': 0.1
        },
    ]
    timesteps_per_level = TOTAL_TIMESTEPS // len(curriculum_levels)

    model: Optional[SAC] = None
    
    env_kwargs = {"perception_model_path": PERCEPTION_MODEL_PATH}

    for level_idx, level_config in enumerate(curriculum_levels):
        print("\n" + "="*50)
        print(f"Starting Curriculum Level {level_idx + 1}/{len(curriculum_levels)}")
        print(f"Configuration: {level_config}")
        print("="*50 + "\n")

        env_kwargs["env_generator"] = ObstacleEnvironmentGenerator(
            random_seed=int(time.time()) + level_idx,
            bezier_n=level_config['bezier_n'],
            band_width=level_config['band_width'],
            obstacle_density=level_config['obstacle_density'],
            x_variation_amplitude=level_config['x_variation_amplitude'],
        )
        # The reuse probability is passed directly to the environment instance, not the generator
        env_kwargs["environment_reuse_probability"] = level_config['environment_reuse_probability']

        train_env = make_vec_env(
            MyShipEnv,
            n_envs=NUM_ENVS,
            seed=int(time.time()) % 2, # Retaining original seed logic
            env_kwargs=env_kwargs,
        )

        if model is None:
            model = SAC(
                "MlpPolicy",
                train_env,
                policy_kwargs=SAC_POLICY_KWARGS,
                learning_rate=SAC_LEARNING_RATE,
                buffer_size=SAC_BUFFER_SIZE,
                learning_starts=SAC_LEARNING_STARTS,
                batch_size=SAC_BATCH_SIZE,
                tau=SAC_TAU,
                gamma=SAC_GAMMA,
                train_freq=SAC_TRAIN_FREQ,
                gradient_steps=SAC_GRADIENT_STEPS,
                tensorboard_log=TENSORBOARD_LOG_DIR,
                verbose=0,
            )
        else:
            model.set_env(train_env)

        callback = ProgressCallback(
            save_interval=SAVE_INTERVAL,
            model_dir=MODEL_SAVE_DIR,
            level_idx=level_idx,
            total_level_steps=timesteps_per_level
        )

        model.learn(
            total_timesteps=timesteps_per_level,
            reset_num_timesteps=False,
            tb_log_name=f"SAC_Level_{level_idx+1}",
            callback=callback
        )
        
        # Save model at the end of each level
        level_model_path = os.path.join(MODEL_SAVE_DIR, f"sac_ship_planner_level_{level_idx + 1}.zip")
        model.save(level_model_path)
        print(f"\nCompleted Level {level_idx + 1}. Model saved to: {level_model_path}")

        train_env.close()

    if model:
        print("\nTraining complete. Saving final model...")
        os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
        model.save(FINAL_MODEL_PATH)
        print(f"Final model saved to: {FINAL_MODEL_PATH}")
    
    print("\n" + "="*50)
    print("TRAINING FINISHED")
    print("="*50)

if __name__ == '__main__':
    if os.name == 'nt':
        multiprocessing.freeze_support()
    main()

# --- END OF FILE training/train_sac_agent.py ---