"""
Utility functions for DreamerV3 training.

Provides logging, visualization, and helper functions for training
and monitoring DreamerV3 agents.
"""

import os
import numpy as np
import torch
import wandb
import imageio


# ============================================================================
# Weights & Biases Logging
# ============================================================================

def log_hparams(config, run_name, resume_id=None):
    """
    Initialize Weights & Biases logging with hyperparameters.
    
    Args:
        config: Training configuration object
        run_name: Name for the wandb run
        resume_id: Optional wandb run ID to resume from
    """
    with open(config.wandb_key, "r", encoding="utf-8") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

    if resume_id:
        # Resume existing run
        wandb.init(
            project="dreamerv3-atari-v2",
            id=resume_id,
            resume="must",
            config=vars(config),
        )
        print(f"Resumed wandb run: {resume_id}")
    else:
        # Start new run
        wandb.init(
            project="dreamerv3-atari-v2",
            name=run_name,
            config=vars(config),
            save_code=True,
        )
    
    return wandb.run.id


def log_losses(step: int, losses: dict):
    """
    Log training losses to Weights & Biases.
    
    Args:
        step: Current training step
        losses: Dictionary containing all loss values
    """
    wandb.log(
        {
            "Loss/World": losses["world_loss"],
            "Loss/Recon": losses["recon_loss"],
            "Loss/Reward": losses["reward_loss"],
            "Loss/Continue": losses["continue_loss"],
            "Loss/KL": losses["kl_loss"],
            "Loss/Actor": losses["actor_loss"],
            "Loss/Critic": losses["critic_loss"],
            "Loss/Retnorm_Scale": losses["retnorm_scale"],
            "Entropy/Actor": losses["actor_entropy"],
            "Entropy/Prior": losses["prior_entropy"],
            "Entropy/Posterior": losses["posterior_entropy"],
        },
        step=step,
    )


def log_rewards(step: int, avg_score: float, best_score: float, 
                mem_size: int, episode: int, total_episodes: int):
    """
    Log reward metrics and print training progress.
    
    Args:
        step: Current training step
        avg_score: Average score over recent episodes
        best_score: Best score achieved so far
        mem_size: Current replay buffer size
        episode: Current episode number
        total_episodes: Total episodes to train for
    """
    wandb.log(
        {
            "Reward/Average": avg_score,
            "Reward/Best": best_score,
            "Memory/Size": mem_size,
        },
        step=step,
    )

    # Print progress to console
    e_str = f"[Ep {episode:05d}/{total_episodes}]"
    a_str = f"Avg.Score = {avg_score:8.2f}"
    b_str = f"Best.Score = {best_score:8.2f}"
    m_str = f"Mem.Size = {mem_size:7d}"
    s_str = f"Step = {step:8d}"
    print(f"{e_str} | {a_str} | {b_str} | {m_str} | {s_str}", end="\r")


# ============================================================================
# Visualization Utilities
# ============================================================================

def _to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to uint8 NumPy array in HWC format.
    
    Useful for preparing images for visualization and logging.
    
    Args:
        t: Tensor in CHW format with values in [0, 1]
        
    Returns:
        NumPy array in HWC format with values in [0, 255]
    """
    t = t.detach().to(dtype=torch.float32, device="cpu").clamp(0, 1)
    arr = (t.numpy() * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    return arr


def log_recon_images(step: int, orig: torch.Tensor, recon: torch.Tensor, 
                     t: int, tag: str = "Recon/Image"):
    """
    Log original and reconstructed images to wandb.
    
    Creates a grid showing original images and their
    reconstructions for visual quality assessment.
    
    Args:
        step: Current training step
        orig: Original images [N, C, H, W]
        recon: Reconstructed images [N, C, H, W]
        t: Timestep index for caption
        tag: Wandb logging tag
    """
    N = orig.shape[0]
    cols = []
    
    for i in range(N):
        up = _to_uint8_hwc(orig[i])
        bottom = _to_uint8_hwc(recon[i])
        cols.append(np.concatenate([up, bottom], axis=0))
    
    grid = np.concatenate(cols, axis=1)
    wandb.log({tag: wandb.Image(grid, caption=f"t={t}, N={N}")}, step=step)


def log_recon_video(step: int,
                    obs_seq: torch.Tensor,
                    recon_seq: torch.Tensor,
                    reward: torch.Tensor, 
                    tag: str = "Recon/Video", 
                    fps: int = 6, 
                    max_frames: int = 64,
                    out_dir: str = "videos/wandb_tmp", 
                    store_as: str = "mp4"):
    """
    Log side-by-side video of original and reconstructed sequences to wandb.
    
    Creates a video showing the temporal quality of reconstructions over
    multiple timesteps.
    
    Args:
        step: Current training step
        obs_seq: Original observation sequence [T, 1, C, H, W]
        recon_seq: Reconstructed sequence [T, 1, C, H, W]
        reward: Reward sequence [T, 1]
        tag: Wandb logging tag
        fps: Frames per second for the video
        max_frames: Maximum number of frames to include
        out_dir: Directory for temporary video files
        store_as: Format to save as ("gif" or "mp4")
    """
    os.makedirs(out_dir, exist_ok=True)

    T = min(int(obs_seq.shape[0]), max_frames)
    frames = []
    
    for tt in range(T):
        L = _to_uint8_hwc(obs_seq[tt, 0])
        R = _to_uint8_hwc(recon_seq[tt, 0])
        
        # Convert grayscale to RGB if needed
        if L.shape[2] == 1:
            L = np.repeat(L, 3, axis=2)
        if R.shape[2] == 1:
            R = np.repeat(R, 3, axis=2)
            
        frames.append(np.ascontiguousarray(np.concatenate([L, R], axis=1)))

    # Save and log video
    safe_tag = tag.replace("/", "_")
    caption = "Reward = {:.2f}".format(reward[:T].mean().item())
    
    if store_as == "mp4":
        path = os.path.join(out_dir, f"{safe_tag}_{step}.mp4")
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        wandb.log({tag: wandb.Video(path, format="mp4", caption=caption)}, step=step)
    else:
        path = os.path.join(out_dir, f"{safe_tag}_{step}.gif")
        imageio.mimsave(path, frames, fps=fps)
        wandb.log({tag: wandb.Video(path, format="gif", caption=caption)}, step=step)