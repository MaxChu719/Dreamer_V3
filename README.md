# DreamerV3 for Atari

A PyTorch implementation of **DreamerV3** for Atari 2600 games, based on the paper ["Mastering Diverse Domains through World Models"](https://arxiv.org/abs/2301.04104) by Hafner et al. (2023).

This implementation is inspired by [naivoder/dreamerv3](https://github.com/naivoder/dreamerv3) and optimized for efficient training on Atari environments.

## 🎮 Overview

DreamerV3 is a model-based reinforcement learning agent that learns a world model to predict future states and rewards, then uses this model to train policies entirely in imagination. This approach enables sample-efficient learning across diverse domains.

### Key Features

- **World Model Learning**: RSSM (Recurrent State-Space Model) with discrete latent states
- **Actor-Critic Training**: Policy optimization in imagined rollouts
- **Parallel Environments**: Vectorized environment execution for faster data collection
- **Advanced Techniques**:
  - Two-hot encoding for continuous value predictions
  - Adaptive gradient clipping
  - Return normalization for stable value learning
  - LAProp optimizer (RMSProp + momentum)
- **Comprehensive Logging**: Weights & Biases integration with video and reconstruction logging

## 🏗️ Architecture

### World Model
- **Encoder**: CNN-based observation encoder (64×64 RGB → 512D embedding)
- **RSSM**: Recurrent state-space model with:
  - Deterministic state: 4096D GRU (Block-wise with 8 blocks)
  - Stochastic state: 32 categorical variables × 32 classes
- **Decoder**: Transposed CNN for observation reconstruction
- **Predictors**: Reward and episode continuation predictors

### Actor-Critic
- **Actor**: 4-layer MLP (512 units) with uniform exploration mixture
- **Critic**: 4-layer MLP (512 units) with two-hot value encoding

## 📦 Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/MaxChu719/Dreamer_V3.git
cd Dreamer_V3

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Weights & Biases Setup

1. Create a [Weights & Biases](https://wandb.ai/) account
2. Save your API key to a file:
   ```bash
   echo "your_wandb_api_key_here" > wandb_key.txt
   ```

## 🚀 Usage

### Basic Training

Train on Pong with default hyperparameters:

```bash
python train.py --env "ALE/Pong-v5" --wandb_key wandb_key.txt
```

### Custom Configuration

```bash
python train.py \
  --env "ALE/Breakout-v5" \
  --num_envs 16 \
  --total_steps 10000000 \
  --lr 4e-5 \
  --actor_lr 4e-5 \
  --critic_lr 4e-5 \
  --wandb_key wandb_key.txt
```

### Resume Training

```bash
python train.py \
  --env "ALE/Pong-v5" \
  --resume checkpoint.pt \
  --resume_wandb_id your_run_id \
  --wandb_key wandb_key.txt
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--env` | Atari environment name | Required |
| `--num_envs` | Number of parallel environments | 16 |
| `--total_steps` | Total environment steps | 1,000,000 |
| `--capacity` | Replay buffer capacity | 1,000,000 |
| `--sequence_length` | Training sequence length | 64 |
| `--imagination_horizon` | Imagination rollout horizon | 16 |
| `--lr` | World model learning rate | 4e-5 |
| `--actor_lr` | Actor learning rate | 4e-5 |
| `--critic_lr` | Critic learning rate | 4e-5 |
| `--discount` | Discount factor γ | 0.997 |
| `--video_interval` | Video logging interval (episodes) | 100 |
| `--wandb_key` | Path to Weights & Biases API key | Required |
| `--resume` | Path to checkpoint for resuming | None |
| `--resume_wandb_id` | Weights & Biases run ID to resume | None |

## 📊 Monitoring

Training metrics are logged to Weights & Biases:

- **Rewards**: Average and best episode scores
- **Losses**: World model, actor, and critic losses
- **Entropy**: Policy and latent state entropy
- **Videos**: Periodic episode recordings
- **Reconstructions**: Original vs. reconstructed observations

Access your dashboard at: `https://wandb.ai/your-username/dreamerv3-atari-v2`

## 🎯 Supported Environments

This implementation supports all **63 Atari 2600 games** available in the Arcade Learning Environment (ALE):

- Adventure, AirRaid, Alien, Amidar, Assault, Asterix, Asteroids, Atlantis
- BankHeist, BattleZone, BeamRider, Berzerk, Bowling, Boxing, Breakout
- Carnival, Centipede, ChopperCommand, CrazyClimber, Defender, DemonAttack
- DoubleDunk, ElevatorAction, Enduro, FishingDerby, Freeway, Frostbite
- Gopher, Gravitar, Hero, IceHockey, Jamesbond, JourneyEscape
- Kangaroo, Krull, KungFuMaster, MontezumaRevenge, MsPacman
- NameThisGame, Phoenix, Pitfall, Pong, Pooyan, PrivateEye
- Qbert, Riverraid, RoadRunner, Robotank, Seaquest, Skiing
- Solaris, SpaceInvaders, StarGunner, Tennis, TimePilot
- Tutankham, UpNDown, Venture, VideoPinball, WizardOfWor
- YarsRevenge, Zaxxon

See `env.py` for the complete list.

## 🧪 Implementation Details

### Key Components

1. **Replay Buffer** (`ReplayBuffer` in `train.py`)
   - Stores sequences of (observation, action, reward, done)
   - Supports efficient random sampling of sequences
   - Handles episode boundaries properly

2. **World Model** (`WorldModel` in `model.py`)
   - Learns to predict next observations and rewards
   - Uses KL balancing for stable latent learning
   - Two-hot encoding for reward prediction

3. **RSSM** (`RSSM` in `model.py`)
   - Block-wise GRU for deterministic state
   - Categorical latent variables for stochastic state
   - Separate prior and posterior networks

4. **Actor-Critic** (`Actor`, `Critic` in `model.py`)
   - Trained on imagined trajectories from world model
   - λ-returns for value estimation
   - Reinforce with baseline for policy gradients

### Preprocessing

- **Frame stacking**: 4 frames with frame skip
- **Resolution**: 64×64 RGB (not grayscale for better feature learning)
- **Normalization**: Pixel values scaled to [0, 1]
- **Action space**: Discrete actions (varies by game)

### Training Process

1. **Data Collection**: Parallel environments collect experience
2. **Replay Buffer**: Store sequences with episode boundaries
3. **World Model Update**: Train on sampled sequences
4. **Imagination**: Generate rollouts from learned world model
5. **Actor-Critic Update**: Optimize policy and value function
6. **Repeat**: Continue until convergence or step limit

## 📝 File Structure

```
dreamerv3-atari/
├── train.py           # Main training loop and replay buffer
├── model.py           # World model, actor, critic, and utilities
├── env.py             # Environment creation and wrappers
├── utils.py           # Logging and visualization utilities
├── requirements.txt   # Python dependencies
├── .gitignore        # Git ignore patterns
└── README.md         # This file
```

## 🔬 Hyperparameters

Default configuration (from DreamerV3 paper):

```python
embed_dim = 512           # Encoder output dimension
latent_dim = 32          # Number of categorical variables
num_classes = 32         # Classes per categorical
deter_dim = 4096         # GRU hidden size
lr = 4e-5                # Learning rate
discount = 0.997         # Discount factor
gae_lambda = 0.95        # GAE parameter
imagination_horizon = 16  # Rollout length
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce `num_envs` (default: 16)
- Reduce `sequence_length` (default: 64)
- Reduce batch size in imagination

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Increase `num_envs` for more parallelism
- Use `torch.backends.cudnn.benchmark = True` (already enabled)

### Video Recording Issues
- Check `video_folder` permissions
- Ensure `imageio` is installed correctly
- Videos are automatically cleaned up after logging to wandb

## 📚 References

- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023)
- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) (Hafner et al., 2020)
- [naivoder/dreamerv3](https://github.com/naivoder/dreamerv3) - Original PyTorch reimplementation

## 📄 License

This project is provided for educational and research purposes. Please refer to the original DreamerV3 paper and implementations for licensing details.

## 🙏 Acknowledgments

- Danijar Hafner and colleagues for the DreamerV3 algorithm
- The naivoder/dreamerv3 implementation for inspiration
- The Arcade Learning Environment (ALE) team
- OpenAI Gymnasium for the environment interface

---

**Note**: This is a research implementation. For production use or commercial applications, please refer to the official DreamerV3 codebase and ensure proper licensing.
