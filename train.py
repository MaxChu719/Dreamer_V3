"""
DreamerV3 Training Script

Main training loop for DreamerV3 on Atari environments.
Implements experience collection, replay buffer management, and model updates.
"""

import os
import argparse
import warnings
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import ale_py
import gymnasium as gym

from model import (
    WorldModel, Actor, Critic, LAProp, 
    TwoHotCategoricalStraightThrough, OneHotCategoricalStraightThrough,
    init_weights, adaptive_gradient_clip, Independent
)
from env import make_vec_env, VideoLoggerWrapper, ENV_LIST
from utils import log_hparams, log_losses, log_rewards, log_recon_images, log_recon_video

warnings.simplefilter("ignore")
gym.register_envs(ale_py)
torch.backends.cudnn.benchmark = True


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DreamerConfig:
    """
    Hyperparameters for DreamerV3 training.
    
    Architecture:
        embed_dim: Observation encoder output dimension
        base_cnn_channels: Base channel count for CNN layers
        latent_dim: Number of categorical variables in stochastic state
        num_classes: Number of classes per categorical variable
        deter_dim: Dimension of deterministic GRU state
        
    Training:
        lr: Learning rate for world model
        actor_lr: Learning rate for actor
        critic_lr: Learning rate for critic
        eps: Epsilon for numerical stability in optimizer
        discount: Discount factor for rewards
        gae_lambda: Lambda parameter for GAE
        entropy_coef: Entropy regularization coefficient
        
    World Model:
        rep_loss_scale: Scale for representation loss
        free_bits: KL divergence lower bound
        
    Replay and Updates:
        capacity: Total replay buffer capacity
        num_envs: Number of parallel environments (Batch size for training)
        sequence_length: Length of sequences for training
        min_buffer_size: Minimum buffer size before training
        update_interval: Update every N environment steps
        updates_per_step: Number of gradient updates per interval
        
    Imagination:
        imagination_horizon: Steps to imagine ahead for actor-critic
        
    Value Learning:
        critic_bins: Number of bins for two-hot value encoding
        critic_ema_decay: EMA decay for target critic
        retnorm_scale: Scale for return normalization
        retnorm_limit: Minimum normalization scale
        retnorm_decay: Decay rate for return normalization
        
    Logging:
        video_interval: Log video every N episodes
        recon_log_interval: Log reconstruction previews every N learner steps
        recon_log_images: Number of samples in reconstruction grid
        recon_log_video_frames: Number of frames for reconstruction video
    """
    
    # Architecture
    embed_dim: int = 512
    base_cnn_channels: int = 32
    latent_dim: int = 32
    num_classes: int = 32
    deter_dim: int = 4096
    
    # Training
    lr: float = 4e-5
    actor_lr: float = 4e-5
    critic_lr: float = 4e-5
    eps: float = 1e-20
    discount: float = 0.997
    gae_lambda: float = 0.95
    entropy_coef: float = 3e-4
    zero_init_state: bool = True
    
    # World Model
    rep_loss_scale: float = 0.1
    free_bits: float = 1.0
    reward_bins: int = 255
    
    # Replay and Updates
    capacity: int = 2_000_000
    num_envs: int = 16
    sequence_length: int = 64
    min_buffer_size: int = 1000
    update_interval: int = 1
    updates_per_step: int = 1
    
    # Imagination
    imagination_horizon: int = 15
    
    # Value Learning
    critic_bins: int = 255
    critic_ema_decay: float = 0.98
    critic_ema_regularizer: float = 1.0
    retnorm_scale: float = 1.0
    retnorm_limit: float = 1.0
    retnorm_decay: float = 0.99
    
    # System
    device: str = "cuda"
    mixed_precision: bool = True
    episodes: int = 100_000

    # Checkpoint
    checkpoint_save_interval_per_episode: int = 100
    
    # Logging
    video_interval: int = 5
    recon_log_interval: int = 500
    recon_log_images: int = 8
    recon_log_video_frames: int = 128
    wandb_key: str = "../wandb.txt"


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Efficient replay buffer for storing environment transitions.
    
    Stores observations as uint8 to save memory, and maintains separate
    buffers for each parallel environment to enable efficient sampling.
    
    Args:
        config: Training configuration
        device: PyTorch device for tensors
        obs_shape: Shape of observations (C, H, W)
    """
    
    def __init__(self, config, device, obs_shape):
        self.num_envs = config.num_envs
        self.capacity = config.capacity // self.num_envs
        self.sequence_length = config.sequence_length
        self.device = device
        self.obs_shape = obs_shape

        # Allocate buffers (using memory-efficient dtypes)
        self.obs_buf = np.zeros((self.num_envs, self.capacity, *obs_shape), dtype=np.float16)
        self.act_buf = np.zeros((self.num_envs, self.capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((self.num_envs, self.capacity), dtype=np.float16)
        self.done_buf = np.zeros((self.num_envs, self.capacity), dtype=np.bool_)
        self.stoch_buf = np.zeros(
            (self.num_envs, self.capacity, config.latent_dim, config.num_classes),
            dtype=np.float16,
        )
        self.deter_buf = np.zeros(
            (self.num_envs, self.capacity, config.deter_dim), dtype=np.float16
        )
        self.pre_act_buf = np.zeros((self.num_envs, self.capacity), dtype=np.uint8)
        
        # Track buffer state
        self.positions = np.zeros(self.num_envs, dtype=np.int64)
        self.full = [False] * self.num_envs

    def store(self, obs, act, rew, done, pre_stoch, pre_deter, pre_act):
        """Store transitions for all environments."""
        for env_idx in range(self.num_envs):
            pos = self.positions[env_idx]
            idx = pos % self.capacity

            self.obs_buf[env_idx, idx] = obs[env_idx]
            self.act_buf[env_idx, idx] = act[env_idx]
            self.rew_buf[env_idx, idx] = rew[env_idx].astype(np.float16)
            self.done_buf[env_idx, idx] = done[env_idx]
            self.stoch_buf[env_idx, idx] = pre_stoch[env_idx].cpu().numpy().astype(np.float16)
            self.deter_buf[env_idx, idx] = pre_deter[env_idx].cpu().numpy().astype(np.float16)
            self.pre_act_buf[env_idx, idx] = pre_act[env_idx].cpu().numpy()

            self.positions[env_idx] += 1
            if self.positions[env_idx] >= self.capacity:
                self.full[env_idx] = True
                self.positions[env_idx] = 0

    def sample(self):
        """
        Sample one sequence from each environment's buffer.
        
        Returns:
            Dictionary containing:
                - stoch: Initial stochastic states
                - deter: Initial deterministic states
                - observation: Observation sequences
                - action: Action sequences
                - reward: Reward sequences
                - done: Termination flags
        """
        indices = []
        start_indices = []
        
        for env_idx in range(self.num_envs):
            current_size = self.capacity if self.full[env_idx] else self.positions[env_idx]
            valid_end = current_size - self.sequence_length

            if valid_end <= 0:
                start = 0
            else:
                start = np.random.randint(0, valid_end)

            env_indices = (start + np.arange(self.sequence_length)) % self.capacity
            indices.append(env_indices)
            start_indices.append(start)

        indices = np.stack(indices)

        return {
            "init_stoch": torch.as_tensor(
                self.stoch_buf[np.arange(self.num_envs), start_indices],
                device=self.device,
                dtype=torch.float32,
            ),
            "init_deter": torch.as_tensor(
                self.deter_buf[np.arange(self.num_envs), start_indices],
                device=self.device,
                dtype=torch.float32,
            ),
            "init_action": torch.as_tensor(
                self.pre_act_buf[np.arange(self.num_envs), start_indices],
                device=self.device,
                dtype=torch.long,
            ),
            "observation": torch.as_tensor(
                self.obs_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.float32,
                device=self.device,
            ).permute(1, 0, 2, 3, 4),
            "action": torch.as_tensor(
                self.act_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.long,
                device=self.device,
            ).permute(1, 0),
            "reward": torch.as_tensor(
                self.rew_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.float32,
                device=self.device,
            ).permute(1, 0),
            "done": torch.as_tensor(
                self.done_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.float32,
                device=self.device,
            ).permute(1, 0),
        }

    def __len__(self):
        """Return minimum available length across all environments."""
        return min(
            pos if not full else self.capacity
            for pos, full in zip(self.positions, self.full)
        )

    def size(self):
        """Return total size of the buffer."""
        return sum(
            pos if not full else self.capacity
            for pos, full in zip(self.positions, self.full)
        )


# ============================================================================
# DreamerV3 Agent
# ============================================================================

class DreamerV3:
    """
    Main DreamerV3 agent implementing world model learning and actor-critic.
    
    The agent maintains:
    - World model (RSSM + encoder/decoder)
    - Actor network (policy)
    - Critic network (value function)
    - Target critic (for stable value learning)
    - Replay buffer for experience
    
    Args:
        obs_shape: Shape of observations (C, H, W)
        action_dim: Number of discrete actions
        config: Training configuration
    """
    
    def __init__(self, obs_shape, action_dim, config):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.device)
        self.num_envs = config.num_envs

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config, self.device, obs_shape)

        # Initialize world model
        self.world_model = WorldModel(
            obs_shape[0],
            action_dim,
            config.embed_dim,
            config.base_cnn_channels,
            config.latent_dim,
            config.num_classes,
            config.deter_dim,
            obs_shape,
            config.reward_bins,
        ).to(self.device)

        # Initialize actor-critic
        feature_dim = config.deter_dim + config.latent_dim * config.num_classes
        self.actor = Actor(feature_dim, action_dim).to(self.device)
        self.critic = Critic(feature_dim, config.critic_bins).to(self.device)
        
        # Initialize target critic
        self.target_critic = Critic(feature_dim, config.critic_bins).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # Apply weight initialization
        self.world_model.apply(init_weights)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        # Initialize optimizers
        self.optimizers = {
            "world": LAProp(
                self.world_model.parameters(),
                lr=config.lr,
                betas=(0.9, 0.99),
                eps=config.eps,
            ),
            "actor": LAProp(
                self.actor.parameters(),
                lr=config.actor_lr,
                betas=(0.9, 0.99),
                eps=config.eps,
            ),
            "critic": LAProp(
                self.critic.parameters(),
                lr=config.critic_lr,
                betas=(0.9, 0.99),
                eps=config.eps,
            ),
        }
        
        # Initialize gradient scalers for mixed precision
        self.scalers = {
            "world": torch.amp.GradScaler("cuda"),
            "actor": torch.amp.GradScaler("cuda"),
            "critic": torch.amp.GradScaler("cuda"),
        }

        # Initialize hidden states
        self.init_hidden_state()
        self._reset_stoch, self._reset_deter, self._reset_action = self.hidden_state
        
        self.gradient_step = 0
        self.env_step = 0
    
    def set_env_step(self, env_step: int):
        """Update environment step counter for logging."""
        self.env_step = env_step
    
    def init_hidden_state(self):
        """Initialize RSSM hidden states for all environments."""
        self.hidden_state = self.world_model.rssm.init_state(
            self.num_envs, self.device
        )

    def reset_hidden_states(self, done_indices):
        """Reset hidden states for environments that terminated."""
        if not done_indices.any():
            return

        stoch, deter, action = self.hidden_state
        stoch[done_indices] = self._reset_stoch[done_indices]
        deter[done_indices] = self._reset_deter[done_indices]
        action[done_indices] = self._reset_action[done_indices]

    def act(self, observations):
        """
        Select actions based on current observations.
        
        Uses the posterior distribution (incorporating observations) to
        update the stochastic state, then samples actions from the actor.
        
        Args:
            observations: NumPy array of observations [num_envs, C, H, W]
            
        Returns:
            actions: NumPy array of selected actions [num_envs]
        """
        obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            pre_stoch, pre_deter, pre_action = self.hidden_state

            # Predict current deterministic state
            _, current_deter = self.world_model.rssm.imagine_step(pre_stoch, pre_deter, pre_action)
            
            # Encode observations
            embed = self.world_model.encoder(obs)

            # Update current stochastic state using posterior
            post_logits = self.world_model.rssm.observe_step(current_deter, embed)
            post_logits = post_logits.view(self.num_envs, self.config.latent_dim, self.config.num_classes)
            current_stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)

            # Sample actions
            feature = torch.cat([current_deter, current_stoch.flatten(1)], dim=1)
            action_dist = self.actor(feature)
            actions = action_dist.sample()

            # Update hidden state
            self.hidden_state = (current_stoch, current_deter, actions)

        return actions.cpu().numpy()

    def store_transition(self, obs, actions, rewards, dones, pre_state):
        """Store environment transitions in replay buffer."""
        pre_sstoch, pre_sdeter, pre_saction = pre_state
        
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device).cpu().numpy()
            
        self.replay_buffer.store(obs_tensor, actions, rewards, dones, pre_sstoch.detach(), pre_sdeter.detach(), pre_saction.detach())

    def update_world_model(self, batch):
        """
        Update world model using reconstruction and prediction losses.
        
        Computes:
        - Reconstruction loss (MSE on decoded observations)
        - Reward prediction loss (two-hot encoding)
        - Continue prediction loss (BCE on episode termination)
        - KL divergence between prior and posterior (dynamics + representation)
        
        Args:
            batch: Dictionary of batched sequences from replay buffer
            
        Returns:
            losses_dict: Dictionary of loss values for logging
            preview: Optional visualization data for wandb
        """
        self.optimizers["world"].zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            init_stoch = batch["init_stoch"] if not self.config.zero_init_state else torch.zeros_like(batch["init_stoch"])
            init_deter = batch["init_deter"] if not self.config.zero_init_state else torch.zeros_like(batch["init_deter"])
            init_action = batch["init_action"] if not self.config.zero_init_state else torch.zeros_like(batch["init_action"])
            obs, actions, dones = batch["observation"], batch["action"], batch["done"]

            # Forward pass through world model
            (priors, posteriors), replay_features, recon_pred, reward_dist, continue_pred, final_state = (
                self.world_model.observe(obs, actions, dones, init_stoch, init_deter, init_action)
            )

            # Compute entropy for monitoring
            prior_entropy = torch.stack([p.entropy() for p in priors]).mean()
            post_entropy = torch.stack([q.entropy() for q in posteriors]).mean()

            # Reconstruction loss
            recon_target = obs.flatten(0, 1)
            recon_loss = F.mse_loss(recon_pred, recon_target, reduction="none").sum(dim=(1, 2, 3)).mean()

            # Reward prediction loss
            reward_loss = -reward_dist.log_prob(batch["reward"].flatten(0, 1)).mean()
            
            # Continue prediction loss
            continue_loss = F.binary_cross_entropy_with_logits(
                continue_pred.flatten(0, 1), (1 - batch["done"].flatten(0, 1))
            )

            # Dynamics loss: KL(posterior || prior) with free bits
            dyn_loss = torch.stack([
                torch.maximum(
                    torch.tensor(self.config.free_bits, device=self.device),
                    torch.distributions.kl_divergence(
                        Independent(
                            OneHotCategoricalStraightThrough(
                                logits=posterior.base_dist.logits.detach()
                            ), 1
                        ),
                        prior,
                    ),
                )
                for prior, posterior in zip(priors, posteriors)
            ]).mean()

            # Representation loss: KL(prior || posterior) with free bits
            rep_loss = torch.stack([
                torch.maximum(
                    torch.tensor(self.config.free_bits, device=self.device),
                    torch.distributions.kl_divergence(
                        posterior,
                        Independent(
                            OneHotCategoricalStraightThrough(
                                logits=prior.base_dist.logits.detach()
                            ), 1
                        ),
                    ),
                )
                for prior, posterior in zip(priors, posteriors)
            ]).mean()

            kl_loss = dyn_loss + rep_loss * self.config.rep_loss_scale
            total_loss = recon_loss + reward_loss + continue_loss + kl_loss

        # Backward pass with gradient scaling
        self.scalers["world"].scale(total_loss).backward()
        self.scalers["world"].unscale_(self.optimizers["world"])
        adaptive_gradient_clip(self.world_model, clip_factor=0.3, eps=1e-3)
        self.scalers["world"].step(self.optimizers["world"])
        self.scalers["world"].update()
    
        losses_dict = {
            "world_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
            "prior_entropy": prior_entropy.item(),
            "posterior_entropy": post_entropy.item(),
        }

        # Generate visualization preview
        preview = None
        if self.gradient_step % self.config.recon_log_interval == 0:
            T, B = actions.size(0), actions.size(1)
            C, H, W = recon_pred.shape[1:]

            recon_vis = recon_pred.clamp(0, 1).view(T, B, C, H, W)

            n = min(self.config.recon_log_images, B)
            t_mid = T // 2
            
            img_orig = obs[t_mid, :n]
            img_reco = recon_vis[t_mid, :n]

            Tvid = min(T, self.config.recon_log_video_frames)
            vid_orig = obs[:Tvid, :1]
            vid_reco = recon_vis[:Tvid, :1]
            reward = batch["reward"][:Tvid, :1]

            preview = {
                "images": (img_orig, img_reco, t_mid),
                "video": (vid_orig, vid_reco, reward),
            }

        return losses_dict, final_state, replay_features, preview

    def update_actor_and_critic(self, final_state, replay_features, replay_batch):
        """
        Update actor and critic using imagined trajectories and replay data.
        
        Process:
        1. Imagine trajectories using current actor
        2. Compute lambda returns from imagined rewards and values (using online critic)
        3. Train critic on both imagined and replay trajectories with target critic regularization
        4. Train actor using advantage estimates
        
        Args:
            final_state: Final state from world model update
            replay_features: Features from replay batch
            replay_batch: Batch from replay buffer
            
        Returns:
            Dictionary of loss values for logging
        """
        B = self.config.num_envs
        init_state = final_state
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            # Imagine trajectories
            features, actions = self.world_model.rssm.imagine(
                init_state, self.actor, self.config.imagination_horizon
            )

            # Predict rewards and continuation
            flat_features = features.flatten(0, 1)
            reward_logits = self.world_model.reward_decoder(flat_features)
            reward_dist = TwoHotCategoricalStraightThrough(reward_logits, bins=self.config.reward_bins)
            rewards = reward_dist.mean.view(features.shape[0], B)

            continue_pred = self.world_model.continue_decoder(flat_features)
            continues = torch.sigmoid(continue_pred).view(features.shape[0], B)
            discounts = self.config.discount * continues

            # Compute values from ONLINE critic (not target!)
            T, B, _ = features.shape
            critic_dist = TwoHotCategoricalStraightThrough(
                self.critic(features.flatten(0, 1)), 
                bins=self.config.critic_bins
            )
            values = critic_dist.mean.view(T, B)

            # Compute lambda returns (GAE)
            lambda_returns = torch.zeros_like(values)
            lambda_returns[-1] = values[-1]
            for t in reversed(range(T - 1)):
                blended = (1 - self.config.gae_lambda) * values[t] + self.config.gae_lambda * lambda_returns[t + 1]
                lambda_returns[t] = rewards[t] + discounts[t] * blended

            # Return normalization
            returns_flat = lambda_returns.flatten()
            current_scale = torch.quantile(returns_flat, 0.95) - torch.quantile(returns_flat, 0.05)
            self.config.retnorm_scale = self.config.retnorm_decay * self.config.retnorm_scale + (1 - self.config.retnorm_decay) * current_scale.item()
            norm_scale = max(self.config.retnorm_limit, self.config.retnorm_scale)

            # Process replay trajectories
            replay_rewards = replay_batch["reward"]
            replay_dones = replay_batch["done"]
            replay_continues = (1 - replay_dones.float()) * self.config.discount

            # Compute replay returns using ONLINE critic
            replay_critic_dist = TwoHotCategoricalStraightThrough(
                self.critic(replay_features.flatten(0, 1)), 
                bins=self.config.critic_bins
            )
            replay_values = replay_critic_dist.mean.view(replay_features.shape[0], B)

            # Compute replay lambda returns
            replay_lambda_returns = torch.zeros_like(replay_values)
            replay_lambda_returns[-1] = replay_values[-1]
            for t in reversed(range(replay_features.shape[0] - 1)):
                blended = (1 - self.config.gae_lambda) * replay_values[t] + self.config.gae_lambda * replay_lambda_returns[t + 1]
                replay_lambda_returns[t] = replay_rewards[t] + replay_continues[t] * blended

            # Compute target critic distributions for regularization
            target_critic_dist = TwoHotCategoricalStraightThrough(
                self.target_critic(features.flatten(0, 1)), 
                bins=self.config.critic_bins
            )

            target_replay_critic_dist = TwoHotCategoricalStraightThrough(
                self.target_critic(replay_features.flatten(0, 1)), 
                bins=self.config.critic_bins
            )

        # Update critic
        self.optimizers["critic"].zero_grad()

        # Compute critic distributions (with gradients)
        critic_dist_imag = TwoHotCategoricalStraightThrough(
            self.critic(features.flatten(0, 1)), 
            bins=self.config.critic_bins
        )
        
        critic_dist_replay = TwoHotCategoricalStraightThrough(
            self.critic(replay_features.flatten(0, 1)), 
            bins=self.config.critic_bins
        )

        # Loss 1: Predict lambda returns (imagination)
        imagination_loss = -critic_dist_imag.log_prob(lambda_returns.flatten(0, 1)).mean()

        # Loss 2: Predict lambda returns (replay)
        replay_loss = -critic_dist_replay.log_prob(replay_lambda_returns.flatten(0, 1)).mean()

        # Loss 3: Regularize towards target critic (imagination)
        target_reg_loss_imag = -critic_dist_imag.log_prob(target_critic_dist.mean.detach()).mean()

        # Loss 4: Regularize towards target critic (replay)
        target_reg_loss_replay = -critic_dist_replay.log_prob(target_replay_critic_dist.mean.detach()).mean()

        # Combine all losses
        # βval = 1 for imagination, βrepval = 0.3 for replay
        # Critic EMA regularizer = 1 (from Table 4)
        total_critic_loss = imagination_loss + 0.3 * replay_loss + self.config.critic_ema_regularizer * (target_reg_loss_imag + 0.3 * target_reg_loss_replay)

        self.scalers["critic"].scale(total_critic_loss).backward()
        self.scalers["critic"].unscale_(self.optimizers["critic"])
        adaptive_gradient_clip(self.critic, clip_factor=0.3, eps=1e-3)
        self.scalers["critic"].step(self.optimizers["critic"])
        self.scalers["critic"].update()

        # Update target critic (EMA)
        with torch.no_grad():
            for online_param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.mul_(self.config.critic_ema_decay).add_(
                    online_param.data, alpha=1 - self.config.critic_ema_decay
                )

        # Update actor
        self.optimizers["actor"].zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # Compute advantages using online critic values
            advantages = (lambda_returns - values).flatten(0, 1) / norm_scale

            # Compute log probs and entropy
            action_dist = self.actor(features.flatten(0, 1))
            log_probs = action_dist.log_prob(actions.flatten(0, 1))
            entropy = action_dist.entropy()

            actor_loss = -(log_probs * advantages.detach() + self.config.entropy_coef * entropy).mean()

        self.scalers["actor"].scale(actor_loss).backward()
        self.scalers["actor"].unscale_(self.optimizers["actor"])
        adaptive_gradient_clip(self.actor, clip_factor=0.3, eps=1e-3)
        self.scalers["actor"].step(self.optimizers["actor"])
        self.scalers["actor"].update()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": total_critic_loss.item(),
            "actor_entropy": entropy.mean().item(),
        }

    def train(self):
        """
        Perform one training iteration.
        
        Samples batches from replay buffer and updates all networks.
        
        Returns:
            Dictionary of losses, or None if buffer too small
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        losses = {
            "world_loss": 0,
            "recon_loss": 0,
            "reward_loss": 0,
            "continue_loss": 0,
            "kl_loss": 0,
            "actor_loss": 0,
            "critic_loss": 0,
            "actor_entropy": 0,
            "prior_entropy": 0,
            "posterior_entropy": 0,
            "retnorm_scale": 0,
        }

        for _ in range(self.config.updates_per_step):
            batch = self.replay_buffer.sample()

            # Update world model
            wm_losses, final_state, replay_features, preview = self.update_world_model(batch)
            for k, v in wm_losses.items():
                losses[k] += v / self.config.updates_per_step

            # Update actor-critic
            ac_losses = self.update_actor_and_critic(final_state, replay_features, batch)
            for k, v in ac_losses.items():
                losses[k] += v / self.config.updates_per_step

            # Log visualizations
            if preview is not None:
                img_orig, img_reco, t_mid = preview["images"]
                log_recon_images(
                    self.env_step, img_orig, img_reco, t_mid, 
                    tag="Recon/Image"
                )
                vid_orig, vid_reco, reward = preview["video"]
                log_recon_video(
                    self.env_step, vid_orig, vid_reco, reward,
                    tag="Recon/Video", fps=12, max_frames=self.config.recon_log_video_frames
                )

            self.gradient_step += 1
        losses["retnorm_scale"] = self.config.retnorm_scale
        return losses

    def save_checkpoint(self, env_name, step_counter, episode_history, 
                   best_score, best_avg, wandb_run_id, config, 
                   checkpoint_type="latest"):
        """
        Save complete training state including models, optimizers, and metadata.
        
        Args:
            env_name: Environment name for filename
            step_counter: Current environment step
            episode_history: List of episode scores
            best_score: Best single episode score
            best_avg: Best average score
            wandb_run_id: Wandb run ID for resuming
            config: Training configuration
            checkpoint_type: Type of checkpoint (latest, best, best_avg, final)
        """
        os.makedirs("weights", exist_ok=True)
        
        checkpoint = {
            # Model states
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            
            # Optimizer states
            "world_optimizer": self.optimizers["world"].state_dict(),
            "actor_optimizer": self.optimizers["actor"].state_dict(),
            "critic_optimizer": self.optimizers["critic"].state_dict(),
            
            # Gradient scaler states
            "world_scaler": self.scalers["world"].state_dict(),
            "actor_scaler": self.scalers["actor"].state_dict(),
            "critic_scaler": self.scalers["critic"].state_dict(),
            
            # Training state
            "gradient_step": self.gradient_step,
            "env_step": step_counter,
            "episode_history": episode_history,
            "best_score": best_score,
            "best_avg": best_avg,
            
            # Config and normalization
            "retnorm_scale": self.config.retnorm_scale,
            "config": vars(config),
            
            # Wandb info
            "wandb_run_id": wandb_run_id,
        }
        
        filename = f"weights/{env_name}_{checkpoint_type}_dreamerv3.pt"
        torch.save(checkpoint, filename)
        # print(f"\nCheckpoint saved: {filename}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load complete training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with training state to resume from
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model states
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        
        # Load optimizer states
        self.optimizers["world"].load_state_dict(checkpoint["world_optimizer"])
        self.optimizers["actor"].load_state_dict(checkpoint["actor_optimizer"])
        self.optimizers["critic"].load_state_dict(checkpoint["critic_optimizer"])
        
        # Load gradient scaler states
        self.scalers["world"].load_state_dict(checkpoint["world_scaler"])
        self.scalers["actor"].load_state_dict(checkpoint["actor_scaler"])
        self.scalers["critic"].load_state_dict(checkpoint["critic_scaler"])
        
        # Load training counters
        self.gradient_step = checkpoint["gradient_step"]
        
        # Load normalization state
        if "retnorm_scale" in checkpoint:
            self.config.retnorm_scale = checkpoint["retnorm_scale"]
        
        print(f"Resumed from gradient_step {self.gradient_step}, env_step {checkpoint['env_step']}")
        
        return {
            "env_step": checkpoint["env_step"],
            "episode_history": checkpoint["episode_history"],
            "best_score": checkpoint["best_score"],
            "best_avg": checkpoint["best_avg"],
            "wandb_run_id": checkpoint.get("wandb_run_id", None),
            "saved_config": checkpoint.get("config", {}),
        }


# ============================================================================
# Main Training Loop
# ============================================================================

def train_dreamer(args):
    """
    Main training function for DreamerV3 with resume support.
    
    Sets up environment, agent, and runs training loop with periodic
    checkpointing and logging. Can resume from previous checkpoint.
    
    Args:
        args: Command-line arguments containing env name, wandb key, and resume path
    """
    config = DreamerConfig()
    config.wandb_key = args.wandb_key

    # Setup environment
    env = make_vec_env(
        args.env, 
        num_envs=config.num_envs, 
        video_interval=config.video_interval
    )
    
    obs_shape = env.single_observation_space.shape
    act_dim = env.single_action_space.n
    save_prefix = args.env.split("/")[-1]
    
    print(f"Environment: {save_prefix}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {act_dim}")

    # Initialize agent
    agent = DreamerV3(obs_shape, act_dim, config)
    
    # Initialize training state
    step_counter = 0
    episode_history = []
    best_score = float("-inf")
    best_avg = float("-inf")
    wandb_run_id = None
    
    # Check for resume
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Warning: Resume checkpoint not found at {args.resume}")
            print("Starting fresh training instead...")
        else:
            # Load checkpoint
            resume_data = agent.load_checkpoint(args.resume)
            
            # Restore training state
            step_counter = resume_data["env_step"]
            episode_history = resume_data["episode_history"]
            best_score = resume_data["best_score"]
            best_avg = resume_data["best_avg"]
            wandb_run_id = resume_data["wandb_run_id"]
            
            # Verify config compatibility (optional)
            saved_config = resume_data.get("saved_config", {})
            if saved_config:
                print(f"Loaded config from checkpoint - verifying compatibility...")
                # You could add config verification here if needed
            
            print(f"Successfully resumed training!")
            print(f"  Episodes completed: {len(episode_history)}")
            print(f"  Best score: {best_score:.2f}")
            print(f"  Best avg: {best_avg:.2f}")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{save_prefix}_{timestamp}"
    wandb_run_id = log_hparams(config, run_name, resume_id=wandb_run_id)
    
    # Wrap env with video logger (needs step_counter reference)
    env = VideoLoggerWrapper(env, "videos", lambda: step_counter)
    
    # Training state
    avg_reward_window = 100
    log_interval_per_step = 10
    episode_scores = np.zeros(config.num_envs)
    avg_score = 0
    avg_losses = {}

    # Initialize environment
    states, _ = env.reset()
    agent.init_hidden_state()

    # Training loop
    while len(episode_history) < config.episodes:
        # Collect experience
        pre_state = agent.hidden_state
        actions = agent.act(states)
        next_states, rewards, terms, truncs, _ = env.step(actions)
        dones = np.logical_or(terms, truncs)
        
        agent.store_transition(states, actions, rewards, dones, pre_state)
        episode_scores += rewards

        # Handle episode terminations
        reset_indices = np.where(dones)[0]
        if len(reset_indices) > 0:
            agent.reset_hidden_states(reset_indices)
            for idx in reset_indices:
                episode_history.append(episode_scores[idx])
                episode_scores[idx] = 0

        step_counter += 1
        agent.set_env_step(step_counter)
        states = next_states

        # Training updates
        if step_counter % config.update_interval == 0:
            losses = agent.train()
            if losses is not None:
                for k, v in losses.items():
                    if k not in avg_losses:
                        avg_losses[k] = [v]
                    else:
                        avg_losses[k].append(v)

        if step_counter % log_interval_per_step == 0:
            # Logging rewards
            avg_score = np.mean(episode_history[-avg_reward_window:]) if episode_history else 0
            mem_size = agent.replay_buffer.size()
            log_rewards(
                step_counter, avg_score, best_score, mem_size,
                len(episode_history), config.episodes
            )

            # Logging losses
            if avg_losses:
                for k in avg_losses:
                    avg_losses[k] = np.mean(avg_losses[k])
                log_losses(step_counter, avg_losses)
                avg_losses = {}

        # Periodic checkpoint saving (every 1000 episodes)
        if len(episode_history) % config.checkpoint_save_interval_per_episode == 0 and len(episode_history) > 0:
            agent.save_checkpoint(
                save_prefix, step_counter, episode_history,
                best_score, best_avg, wandb_run_id, config,
                checkpoint_type="latest"
            )

        # Save best checkpoints
        if episode_history and max(episode_history) > best_score:
            best_score = max(episode_history)
            agent.save_checkpoint(
                save_prefix, step_counter, episode_history,
                best_score, best_avg, wandb_run_id, config,
                checkpoint_type="best"
            )

        if avg_score > best_avg:
            best_avg = avg_score
            agent.save_checkpoint(
                save_prefix, step_counter, episode_history,
                best_score, best_avg, wandb_run_id, config,
                checkpoint_type="best_avg"
            )

    print(f"\n✓ Training complete! Best average score: {best_avg:.2f}")
    agent.save_checkpoint(
        save_prefix, step_counter, episode_history,
        best_score, best_avg, wandb_run_id, config,
        checkpoint_type="final"
    )
    env.close()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train DreamerV3 on Atari environments"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default=None,
        help="Atari environment name (e.g., ALE/Pong-v5)"
    )
    parser.add_argument(
        "--wandb_key", 
        type=str, 
        default="../wandb.txt",
        help="Path to file containing Weights & Biases API key"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    args = parser.parse_args()

    # Create necessary directories
    for folder in ["videos", "weights"]:
        os.makedirs(folder, exist_ok=True)

    if args.env:
        # Train on single environment
        train_dreamer(args)
    else:
        # Train on all environments in random order
        if args.resume:
            print("Warning: --resume only works with --env specified")
            print("Starting fresh training on all environments")
            args.resume = None
        
        rand_order = np.random.permutation(ENV_LIST)
        for env in rand_order:
            args.env = env
            args.resume = None  # Don't resume for batch training
            train_dreamer(args)


if __name__ == "__main__":
    main()