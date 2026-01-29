"""
DreamerV3 Model Components

This module contains the neural network architectures for DreamerV3:
- World Model (RSSM, Encoder, Decoder)
- Actor-Critic networks
- Custom optimizers and distributions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent


# ============================================================================
# Custom Optimizer
# ============================================================================

class LAProp(torch.optim.Optimizer):
    """
    LAProp optimizer combining RMSProp with momentum.
    
    This optimizer uses adaptive learning rates (like RMSProp) combined with
    momentum updates for better convergence in deep RL settings.
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for momentum and RMSProp (default: (0.9, 0.99))
        eps: Small constant for numerical stability (default: 1e-20)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-20):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.state["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["momentum_buffer"] = torch.zeros_like(p)

                exp_avg_sq = state["exp_avg_sq"]
                momentum_buffer = state["momentum_buffer"]

                # RMSProp-style variance normalization
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                normalized_grad = grad / denom

                # Momentum update
                momentum_buffer.mul_(beta1).add_(normalized_grad, alpha=1 - beta1)

                # Parameter update
                p.add_(momentum_buffer, alpha=-group["lr"])

        return loss


# ============================================================================
# Custom Layers and Utilities
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient alternative to LayerNorm that normalizes based on RMS
    rather than mean and variance.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class BlockGRU(nn.Module):
    """
    Block-wise GRU for efficient parallel processing.
    
    Splits the hidden state into independent blocks, each processed by its
    own GRU cell. This enables better parallelization and scaling.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Total hidden dimension (must be divisible by num_blocks)
        num_blocks: Number of independent GRU blocks
    """
    
    def __init__(self, input_dim, hidden_dim, num_blocks=8):
        super().__init__()
        assert hidden_dim % num_blocks == 0, "hidden_dim must be divisible by num_blocks"
        
        self.num_blocks = num_blocks
        self.block_size = hidden_dim // num_blocks
        
        self.blocks = nn.ModuleList([
            nn.GRUCell(input_dim, self.block_size) 
            for _ in range(num_blocks)
        ])
    
    def forward(self, input, hidden):
        """Process input through parallel GRU blocks."""
        h_blocks = hidden.chunk(self.num_blocks, dim=-1)
        new_blocks = [
            block(input, h_block) 
            for block, h_block in zip(self.blocks, h_blocks)
        ]
        return torch.cat(new_blocks, dim=-1)


# ============================================================================
# Custom Distributions
# ============================================================================

class OneHotCategoricalStraightThrough(torch.distributions.Categorical):
    """
    Categorical distribution with straight-through gradient estimator.
    
    Samples discrete actions but allows gradients to flow through the
    continuous probabilities via the straight-through estimator trick.
    """
    
    def sample(self, sample_shape=torch.Size()):
        # Get discrete indices
        indices = super().sample(sample_shape)
        
        # Convert to one-hot encoding
        one_hot = F.one_hot(indices, self.probs.shape[-1]).to(self.probs)
        
        # Straight-through estimator: forward uses one_hot, backward uses probs
        return self.probs + (one_hot - self.probs).detach()


class TwoHotCategoricalStraightThrough(torch.distributions.Distribution):
    """
    Two-hot encoding distribution for continuous values.
    
    Represents continuous values as a mixture of two adjacent bins in a
    discrete distribution. Uses symlog/symexp transformations for better
    handling of large value ranges.
    
    Args:
        logits: Categorical logits over bins
        bins: Number of discrete bins
        low: Lower bound of value range (in symlog space)
        high: Upper bound of value range (in symlog space)
    """
    
    def __init__(self, logits, bins=255, low=-20.0, high=20.0):
        super().__init__(validate_args=False)
        self.logits = logits
        self.bins = bins
        self.bin_centers = torch.linspace(low, high, bins, device=logits.device)

    def log_prob(self, value):
        """Compute log probability using two-hot interpolation."""
        # Transform value to symlog space
        value = symlog(value).clamp(self.bin_centers[0], self.bin_centers[-1])
        
        # Find continuous bin index
        step = self.bin_centers[1] - self.bin_centers[0]
        indices = (value - self.bin_centers[0]) / step
        indices = indices.clamp(0, self.bins - 1)
        
        # Get lower and upper bins
        lower = indices.floor().long()
        upper = indices.ceil().long()
        alpha = indices - lower.float()

        # Compute weighted log probabilities
        log_probs = F.log_softmax(self.logits, dim=-1)
        lp_lower = log_probs.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        lp_upper = log_probs.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
        
        return (1 - alpha) * lp_lower + alpha * lp_upper

    def sample(self, sample_shape=torch.Size()):
        """Sample using straight-through estimator."""
        # Hard sample from categorical
        indices = torch.distributions.Categorical(logits=self.logits).sample(sample_shape)
        values = self.bin_centers[indices]
        
        # Straight-through: forward uses hard sample, backward uses mean
        mean = self.mean
        return mean + (values - mean).detach()

    @property
    def mean(self):
        """Expected value in original space."""
        probs = F.softmax(self.logits, dim=-1)
        expected_symlog = (probs * self.bin_centers).sum(-1)
        return symexp(expected_symlog)


# ============================================================================
# Symlog Transformations
# ============================================================================

def symlog(x):
    """Symmetric log transform for handling large value ranges."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    """Inverse of symlog transformation."""
    return torch.sign(x) * (torch.exp(torch.clamp(torch.abs(x), max=20.0)) - 1.0)


# ============================================================================
# Encoder/Decoder Networks
# ============================================================================

class ObservationEncoder(nn.Module):
    """
    CNN encoder for visual observations.
    
    Encodes 64x64 RGB images into compact latent representations using
    a series of convolutional layers with stride-2 downsampling.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Dimension of output embedding
        base_cnn_channels: Base number of CNN channels (doubles each layer)
    """
    
    def __init__(self, in_channels=3, embed_dim=1024, base_cnn_channels=32):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_cnn_channels, 4, 2),
            nn.BatchNorm2d(base_cnn_channels),
            nn.SiLU(),
            nn.Conv2d(base_cnn_channels, base_cnn_channels * 2, 4, 2),
            nn.BatchNorm2d(base_cnn_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_cnn_channels * 2, base_cnn_channels * 4, 4, 2),
            nn.BatchNorm2d(base_cnn_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_cnn_channels * 4, base_cnn_channels * 8, 4, 2),
            nn.BatchNorm2d(base_cnn_channels * 8),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(base_cnn_channels * 8 * 2 * 2, embed_dim),
            RMSNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.conv, x)


class ObservationDecoder(nn.Module):
    """
    CNN decoder for reconstructing observations.
    
    Reconstructs 64x64 RGB images from latent features using transposed
    convolutions for upsampling.
    
    Args:
        feature_dim: Dimension of input features
        out_channels: Number of output channels (3 for RGB)
        output_size: Spatial size of output (H, W)
        base_cnn_channels: Base number of CNN channels
    """
    
    def __init__(self, feature_dim, out_channels=3, output_size=(64, 64), 
                 base_cnn_channels=32):
        super().__init__()
        self.out_channels = out_channels
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(feature_dim, base_cnn_channels * 8 * 8 * 8),
            RMSNorm(base_cnn_channels * 8 * 8 * 8),
            nn.SiLU(),
            nn.Unflatten(1, (base_cnn_channels * 8, 8, 8)),
            nn.ConvTranspose2d(base_cnn_channels * 8, base_cnn_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_cnn_channels * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(base_cnn_channels * 4, base_cnn_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_cnn_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_cnn_channels * 2, base_cnn_channels, 4, 2, 1),
            nn.BatchNorm2d(base_cnn_channels),
            nn.SiLU(),
            nn.Conv2d(base_cnn_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.net, x)


# ============================================================================
# RSSM (Recurrent State-Space Model)
# ============================================================================

class RSSM(nn.Module):
    """
    Recurrent State-Space Model - the core of DreamerV3's world model.
    
    Maintains two types of state:
    - Deterministic state (deter): Processed by recurrent network
    - Stochastic state (stoch): Sampled from learned distributions
    
    The model learns:
    - Prior p(s_t | h_t): Prediction from deterministic state alone
    - Posterior q(s_t | h_t, o_t): Incorporating actual observation
    
    Args:
        action_dim: Number of discrete actions
        latent_dim: Number of categorical distributions in stochastic state
        num_classes: Number of classes per categorical distribution
        deter_dim: Dimension of deterministic GRU state
        embed_dim: Dimension of observation embeddings
    """
    
    def __init__(self, action_dim, latent_dim, num_classes, deter_dim, embed_dim):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim

        # Prior network: p(s_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim * num_classes),
        )

        # Posterior network: q(s_t | h_t, o_t)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim * num_classes),
        )

        # Recurrent state transition
        self.gru = BlockGRU(latent_dim * num_classes + action_dim, deter_dim)

    def init_state(self, batch_size, device):
        """Initialize RSSM state for a batch."""
        stoch = F.one_hot(torch.zeros(batch_size, self.latent_dim, dtype=torch.long), self.num_classes).float().to(device)
        deter = torch.zeros(batch_size, self.deter_dim).float().to(device)
        action = torch.zeros(batch_size).long().to(device)
        return (stoch, deter, action)
    
    def prior_net_logits_with_mixture_and_reshape(self, deter):
        """
        Compute prior logits with uniform mixture for exploration.
        
        Adds 1% uniform noise to prevent overconfident predictions.
        """
        prior_logits = self.prior_net(deter).view(-1, self.latent_dim, self.num_classes)
        prior_logits = prior_logits - torch.logsumexp(prior_logits, -1, keepdim=True)
        prior_logits = torch.log(
            0.99 * torch.softmax(prior_logits, -1) + 0.01 / self.num_classes
        )
        return prior_logits

    def imagine_step(self, stoch, deter, action):
        """
        One step of imagination using the prior.
        
        Args:
            stoch: Current stochastic state
            deter: Current deterministic state
            action: Action to take
            
        Returns:
            next_stoch, next_deter: Next states
        """
        action_oh = F.one_hot(action, self.action_dim).float()
        gru_input = torch.cat([stoch.flatten(1), action_oh], dim=1)
        
        # Update deterministic state
        deter = self.gru(gru_input, deter)
        
        # Sample stochastic state from prior
        prior_logits = self.prior_net_logits_with_mixture_and_reshape(deter)
        stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        
        return stoch, deter

    def observe_step(self, deter, embed):
        """
        Compute posterior distribution given observation.
        
        Args:
            deter: Deterministic state
            embed: Observation embedding
            
        Returns:
            post_logits: Posterior logits for stochastic state
        """
        post_logits = self.post_net(torch.cat([deter, embed], dim=1))
        post_logits = post_logits.view(-1, self.latent_dim, self.num_classes)
        post_logits = post_logits - torch.logsumexp(post_logits, -1, keepdim=True)
        post_logits = torch.log(
            0.99 * torch.softmax(post_logits, -1) + 0.01 / self.num_classes
        )
        return post_logits

    def imagine(self, init_state, actor, horizon):
        """
        Roll out imagined trajectories using the actor.
        
        Args:
            init_state: Initial (stoch, deter, action) state
            actor: Policy network
            horizon: Number of steps to imagine
            
        Returns:
            features: Imagined feature sequences [T, B, feature_dim]
            actions: Sampled actions [T, B]
        """
        pre_stoch, pre_deter, pre_action = init_state
        features, actions = [], []

        for _ in range(horizon):
            # One imagination step
            stoch, deter = self.imagine_step(pre_stoch, pre_deter, pre_action)

            # Compute feature and sample action
            feature = torch.cat([deter, stoch.flatten(1)], dim=1)
            with torch.no_grad():
                action = actor(feature).sample()

            # Store results
            features.append(feature)
            actions.append(action)

            # Update for next step
            pre_stoch = stoch
            pre_deter = deter
            pre_action = action

        return torch.stack(features), torch.stack(actions)


# ============================================================================
# World Model
# ============================================================================

class WorldModel(nn.Module):
    """
    Complete world model combining RSSM with encoder/decoder networks.
    
    Learns to:
    - Encode observations to latent embeddings
    - Maintain latent state dynamics (RSSM)
    - Decode states back to observations
    - Predict rewards and episode termination
    
    Args:
        in_channels: Number of input image channels
        action_dim: Number of discrete actions
        embed_dim: Dimension of observation embeddings
        base_cnn_channels: Base CNN channel count
        latent_dim: Number of stochastic state categoricals
        num_classes: Classes per categorical
        deter_dim: Deterministic state dimension
        obs_size: Full observation shape (C, H, W)
    """
    
    def __init__(self, in_channels, action_dim, embed_dim, base_cnn_channels,
                 latent_dim, num_classes, deter_dim, obs_size, reward_bins):
        super().__init__()
        
        self.encoder = ObservationEncoder(in_channels, embed_dim, base_cnn_channels)
        self.rssm = RSSM(action_dim, latent_dim, num_classes, deter_dim, embed_dim)
        
        feature_dim = deter_dim + latent_dim * num_classes
        self.decoder = ObservationDecoder(feature_dim, in_channels, obs_size[1:], base_cnn_channels)
        self.reward_decoder = nn.Linear(feature_dim, reward_bins)
        self.continue_decoder = nn.Linear(feature_dim, 1)

        # Initialize reward decoder to zero for stability
        self.reward_decoder.weight.data.zero_()
        self.reward_decoder.bias.data.zero_()

    def observe(self, observations, actions, dones, init_stoch, init_deter, init_action):
        """
        Process observation sequence through world model.
        
        Args:
            observations: Image sequence [T, B, C, H, W]
            actions: Action sequence [T, B]
            dones: Done flags [T, B]
            init_stoch, init_deter: Initial RSSM states at start [T, B, ...]
            init_action: Initial actions [T, B]
            
        Returns:
            (priors, posteriors): Distribution lists for KL loss
            features: State feature sequence
            recon_pred: Reconstructed observations
            reward_dist: Predicted reward distribution
            continue_pred: Episode continuation predictions
        """
        # Encode all observations
        embed = self.encoder(observations.flatten(0, 1))
        embed = embed.view(actions.size(0), actions.size(1), -1)
        
        actions_onehot = F.one_hot(actions, self.rssm.action_dim).float()

        priors, posteriors = [], []
        features = []

        pre_deter = init_deter
        pre_stoch = init_stoch
        pre_action_onehot = F.one_hot(init_action, self.rssm.action_dim).float()

        for t in range(actions.size(0)):
            if t > 0:
                # Create a mask: 0.0 if the previous state was terminal, else 1.0
                mask = 1.0 - dones[t - 1].float()
            
                # Reshape mask to [Batch, 1, 1] for the 3D stochastic state
                pre_stoch = pre_stoch * mask.view(-1, 1, 1)
                # Reshape mask to [Batch, 1] for the 2D deterministic state
                pre_deter = pre_deter * mask.view(-1, 1)
                # Reshape mask to [Batch, 1] for the action one-hot
                pre_action_onehot = pre_action_onehot * mask.view(-1, 1)

            # Update deterministic state
            deter = self.rssm.gru(torch.cat([pre_stoch.flatten(1), pre_action_onehot], dim=1), pre_deter)

            # Sample stochastic state (posterior)
            post_logits = self.rssm.observe_step(deter, embed[t])
            post_dist = Independent(OneHotCategoricalStraightThrough(logits=post_logits), 1)
            stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)

            # Compute prior distribution
            prior_logits = self.rssm.prior_net_logits_with_mixture_and_reshape(deter)
            prior_dist = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)

            feature = torch.cat([deter, stoch.flatten(1)], dim=1)
            features.append(feature)
            priors.append(prior_dist)
            posteriors.append(post_dist)

            # Update for next step
            pre_deter = deter
            pre_stoch = stoch
            pre_action_onehot = actions_onehot[t]
            pre_action = actions[t]

        features = torch.stack(features)
        priors = priors
        posteriors = posteriors
        
        # Decode predictions
        flat_features = features.flatten(0, 1)
        recon_pred = self.decoder(flat_features)
        reward_dist = TwoHotCategoricalStraightThrough(self.reward_decoder(flat_features))
        continue_pred = self.continue_decoder(flat_features)
        final_state = (pre_stoch.detach(), pre_deter.detach(), pre_action.detach())

        return (priors, posteriors), features.detach(), recon_pred, reward_dist, continue_pred, final_state


# ============================================================================
# Actor and Critic Networks
# ============================================================================

class Actor(nn.Module):
    """
    Policy network for action selection.
    
    Maps state features to action probabilities using a categorical
    distribution with uniform mixture for exploration.
    
    Args:
        feature_dim: Dimension of input state features
        action_dim: Number of discrete actions
    """
    
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        """
        Compute action distribution from features.
        
        Returns:
            Categorical distribution over actions with 1% uniform mixture
        """
        logits = self.net(x)
        
        # Add uniform mixture for exploration
        logits = logits - torch.logsumexp(logits, -1, keepdim=True)
        logits = torch.log(
            0.99 * torch.softmax(logits, -1) + 0.01 / self.action_dim
        )
        
        return Categorical(logits=logits)


class Critic(nn.Module):
    """
    Value network for estimating state values.
    
    Uses two-hot encoding to represent continuous value predictions
    as a distribution over discrete bins.
    
    Args:
        feature_dim: Dimension of input state features
        critic_bins: Number of discrete value bins
    """
    
    def __init__(self, feature_dim, critic_bins=255):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, critic_bins),
        )
        
        # Initialize last layer to zeros for stability
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        """Return logits over value bins."""
        return self.net(x)


# ============================================================================
# Initialization Utilities
# ============================================================================

def init_weights(m):
    """
    Initialize network weights with orthogonal initialization.
    
    Uses orthogonal initialization for Conv2d,
    ConvTranspose2d, and Linear layers.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def adaptive_gradient_clip(model, clip_factor=0.3, eps=1e-3):
    """
    Apply adaptive gradient clipping based on parameter norms.
    
    Clips gradients relative to the norm of the parameters themselves,
    preventing gradients from being too large relative to the weights.
    
    Args:
        model: Neural network model
        clip_factor: Clipping threshold as fraction of weight norm
        eps: Small constant for numerical stability
    """
    for param in model.parameters():
        if param.grad is not None:
            weight_norm = torch.norm(param.detach(), p=2)
            grad_norm = torch.norm(param.grad.detach(), p=2)
            max_norm = clip_factor * weight_norm + eps
            
            if grad_norm > max_norm:
                scale = max_norm / (grad_norm + 1e-8)
                param.grad.mul_(scale)