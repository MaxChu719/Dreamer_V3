"""
Environment utilities for DreamerV3 training on Atari games.

Provides environment creation, preprocessing, and vectorization utilities
for Atari 2600 games using the Arcade Learning Environment (ALE).
"""

import os
import time
import numpy as np
import gymnasium as gym
import wandb


# ============================================================================
# Atari Environment List
# ============================================================================

ENV_LIST = [
    "ALE/Adventure-v5",
    "ALE/AirRaid-v5",
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
    "ALE/Assault-v5",
    "ALE/Asterix-v5",
    "ALE/Asteroids-v5",
    "ALE/Atlantis-v5",
    "ALE/BankHeist-v5",
    "ALE/BattleZone-v5",
    "ALE/BeamRider-v5",
    "ALE/Berzerk-v5",
    "ALE/Bowling-v5",
    "ALE/Boxing-v5",
    "ALE/Breakout-v5",
    "ALE/Carnival-v5",
    "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/Defender-v5",
    "ALE/DemonAttack-v5",
    "ALE/DoubleDunk-v5",
    "ALE/ElevatorAction-v5",
    "ALE/Enduro-v5",
    "ALE/FishingDerby-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Gravitar-v5",
    "ALE/Hero-v5",
    "ALE/IceHockey-v5",
    "ALE/Jamesbond-v5",
    "ALE/JourneyEscape-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/MsPacman-v5",
    "ALE/NameThisGame-v5",
    "ALE/Phoenix-v5",
    "ALE/Pitfall-v5",
    "ALE/Pong-v5",
    "ALE/Pooyan-v5",
    "ALE/PrivateEye-v5",
    "ALE/Qbert-v5",
    "ALE/Riverraid-v5",
    "ALE/RoadRunner-v5",
    "ALE/Robotank-v5",
    "ALE/Seaquest-v5",
    "ALE/Skiing-v5",
    "ALE/Solaris-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/StarGunner-v5",
    "ALE/Tennis-v5",
    "ALE/TimePilot-v5",
    "ALE/Tutankham-v5",
    "ALE/UpNDown-v5",
    "ALE/Venture-v5",
    "ALE/VideoPinball-v5",
    "ALE/WizardOfWor-v5",
    "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5",
]


# ============================================================================
# Environment Wrappers
# ============================================================================

class FireOnReset(gym.Wrapper):
    """
    Automatically press FIRE on environment reset for games that require it.
    
    Some Atari games (like Breakout) require pressing FIRE to start.
    This wrapper ensures the game starts immediately after reset.
    """
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        action_meanings = self.env.unwrapped.get_action_meanings()
        
        if "FIRE" in action_meanings:
            # Action index 1 is typically FIRE in ALE
            obs, _, terminated, truncated, _ = self.env.step(1)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
                
        return obs, info


class VideoLoggerWrapper(gym.vector.VectorWrapper):
    """
    Wrapper for logging episode videos to Weights & Biases.
    
    Monitors the video folder for new recordings and automatically
    uploads them to wandb, then removes the local files.
    
    Args:
        env: Vectorized environment to wrap
        video_folder: Directory containing recorded videos
        get_step_callback: Function that returns current training step
    """
    
    def __init__(self, env, video_folder, get_step_callback):
        super().__init__(env)
        self.video_folder = video_folder
        self.last_logged = 0
        self.get_step = get_step_callback

    def step(self, action):
        obs, rewards, terminated, truncated, infos = super().step(action)
        current_step = self.get_step()

        # Find new video files
        new_videos = [
            f for f in os.listdir(self.video_folder)
            if f.endswith(".mp4") and 
            os.path.getmtime(os.path.join(self.video_folder, f)) > self.last_logged
        ]

        # Log and remove videos
        for video_file in sorted(
            new_videos,
            key=lambda x: os.path.getctime(os.path.join(self.video_folder, x))
        ):
            video_path = os.path.join(self.video_folder, video_file)
            wandb.log({"video": wandb.Video(video_path, format="mp4")}, step=current_step)
            os.remove(video_path)
            self.last_logged = time.time()

        return obs, rewards, terminated, truncated, infos


# ============================================================================
# Environment Creation
# ============================================================================

def make_env(env_name, record_video=False, video_folder="videos", 
             video_interval=100, test=False, fire_on_reset=False):
    """
    Create a single Atari environment with standard preprocessing.
    
    Applies:
    - FireOnReset wrapper for games requiring FIRE action
    - AtariPreprocessing (grayscale conversion, frame resizing)
    - Observation transposition to channel-first format
    - Optional video recording
    
    Args:
        env_name: Atari environment name (e.g., "ALE/Pong-v5")
        record_video: Whether to record episode videos
        video_folder: Directory for saving videos
        video_interval: Record every N episodes
        test: If True, disable random no-ops for deterministic evaluation
        
    Returns:
        Configured Gymnasium environment
    """
    # Create base environment
    env = gym.make(env_name, frameskip=1, render_mode="rgb_array" if record_video else None)
    
    if fire_on_reset:
        # Add FIRE on reset for applicable games
        env = FireOnReset(env)
    
    # Standard Atari preprocessing
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=64,
        grayscale_obs=False,  # Keep RGB for better feature learning
        scale_obs=True,       # Normalize to [0, 1]
        noop_max=0 if test else 30,  # Random no-ops at start
    )
    
    # Transpose observations to channel-first (C, H, W)
    env = gym.wrappers.TransformObservation(
        env, 
        lambda obs: np.transpose(obs, (2, 0, 1)), 
        None
    )
    
    # Update observation space to reflect channel-first format
    env.observation_space = gym.spaces.Box(
        low=0, high=1, shape=(3, 64, 64), dtype=np.float32
    )

    # Add video recording if requested
    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: x % video_interval == 0,
            name_prefix=env_name.split("/")[-1],
        )

    return env


def make_vec_env(env_name, num_envs=16, video_folder="videos", video_interval=100):
    """
    Create a vectorized Atari environment for parallel training.
    
    Creates multiple environment instances running in parallel using
    AsyncVectorEnv. Only the first environment records videos to avoid
    redundancy and reduce disk usage.
    
    Args:
        env_name: Atari environment name
        num_envs: Number of parallel environments
        video_folder: Directory for saving videos
        video_interval: Record every N episodes (first env only)
        
    Returns:
        AsyncVectorEnv with num_envs parallel environments
    """
    os.makedirs(video_folder, exist_ok=True)

    # Create environment factories
    # Only first environment records videos
    env_fns = [
        lambda i=i: make_env(
            env_name,
            record_video=(i == 0),
            video_folder=video_folder,
            video_interval=video_interval,
        )
        for i in range(num_envs)
    ]

    vec_env = gym.vector.AsyncVectorEnv(env_fns)
    return vec_env
