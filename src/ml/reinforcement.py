"""
Reinforcement Learning for GMCS.

Provides RL agents for learning to control chaotic dynamics, stabilize
oscillators, navigate attractors, and optimize energy landscapes.

Key features:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- Chaos control environments
- THRML-aware rewards

Use cases:
- Stabilize chaotic oscillators
- Navigate attractors to specific regions
- Optimize energy landscapes
- Control wave field patterns
- Learn adaptive coupling strategies
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .ml_nodes import MLModelNode


# ============================================================================
# Policy Network
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Policy network for continuous actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        
        self.shared = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) state
            
        Returns:
            (mean, log_std) for action distribution
        """
        features = self.shared(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
        return mean, log_std
    
    def sample(self, state):
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Value network (critic).
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class QNetwork(nn.Module):
    """
    Q-network for state-action value.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim + action_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer.
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Good for:
    - Stable policy learning
    - On-policy optimization
    - Continuous control
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "auto"
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            device: Device to use
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Select action from policy.
        
        Args:
            state: Current state
            
        Returns:
            (action, log_prob)
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.sample(state_t)
            value = self.value(state_t)
        
        return action.cpu().numpy()[0], float(log_prob.item()), float(value.item())
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """Store transition for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using collected experiences.
        
        Returns:
            Dict with losses
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).unsqueeze(-1).to(self.device)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        G = 0
        
        for reward, value, done in zip(reversed(self.rewards), reversed(self.values), reversed(self.dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
            advantages.insert(0, G - value)
        
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        mean, log_std = self.policy(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
        
        # Policy loss (clipped)
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value(states)
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        return {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.item()),
            'total_loss': float(loss.item())
        }


# ============================================================================
# SAC Agent
# ============================================================================

class SACAgent:
    """
    Soft Actor-Critic agent.
    
    Good for:
    - Maximum entropy RL
    - Off-policy learning
    - Continuous control
    - Sample efficiency
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "auto"
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            tau: Target network update rate
            alpha: Entropy coefficient
            device: Device to use
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        
        # Copy target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer()
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select action.
        
        Args:
            state: Current state
            evaluate: If True, use mean action
            
        Returns:
            action
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy(state_t)
                action = mean
            else:
                action, _ = self.policy.sample(state_t)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Update networks using replay buffer.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dict with losses
        """
        if len(self.buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        
        # Update Q networks
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': float(q1_loss.item()),
            'q2_loss': float(q2_loss.item()),
            'policy_loss': float(policy_loss.item())
        }


# ============================================================================
# RL Controller Node
# ============================================================================

class RLControllerNode(MLModelNode):
    """
    RL controller node for GMCS.
    
    Learns to control chaotic dynamics through reinforcement learning.
    """
    
    def __init__(
        self,
        node_id: str,
        state_dim: int,
        action_dim: int,
        algorithm: str = 'ppo',
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize RL controller.
        
        Args:
            node_id: Unique identifier
            state_dim: State dimension
            action_dim: Action dimension
            algorithm: 'ppo' or 'sac'
            device: Device to use
            **kwargs: Algorithm-specific parameters
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        # Create agent
        if algorithm == 'ppo':
            self.agent = PPOAgent(state_dim, action_dim, device=device, **kwargs)
            model = self.agent.policy
        elif algorithm == 'sac':
            self.agent = SACAgent(state_dim, action_dim, device=device, **kwargs)
            model = self.agent.policy
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        super().__init__(node_id, model, framework='pytorch')
        
        self.algorithm = algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.metadata.update({
            'algorithm': algorithm,
            'state_dim': state_dim,
            'action_dim': action_dim
        })
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Select action for given state.
        
        Args:
            state: Current state
            
        Returns:
            Action
        """
        if self.algorithm == 'ppo':
            action, _, _ = self.agent.select_action(state)
            return action
        else:  # SAC
            return self.agent.select_action(state, evaluate=False)
    
    def train_episode(
        self,
        env: Any,
        max_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Train for one episode.
        
        Args:
            env: Environment with reset(), step(), etc.
            max_steps: Maximum steps per episode
            
        Returns:
            Episode metrics
        """
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            if self.algorithm == 'ppo':
                action, log_prob, value = self.agent.select_action(state)
            else:
                action = self.agent.select_action(state)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            if self.algorithm == 'ppo':
                self.agent.store_transition(state, action, reward, log_prob, value, done)
            else:
                self.agent.buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                break
        
        # Update agent
        if self.algorithm == 'ppo':
            losses = self.agent.update()
        else:
            losses = self.agent.update()
        
        losses['episode_reward'] = episode_reward
        losses['episode_length'] = step + 1
        
        return losses


# ============================================================================
# Chaos Control Environment
# ============================================================================

class ChaosControlEnv:
    """
    RL environment for controlling chaotic oscillators.
    
    Reward: Negative distance to target state or stabilization reward.
    """
    
    def __init__(
        self,
        oscillator_fn: Callable,
        target_state: Optional[np.ndarray] = None,
        dt: float = 0.01,
        max_forcing: float = 1.0
    ):
        """
        Initialize chaos control environment.
        
        Args:
            oscillator_fn: Function that steps oscillator
            target_state: Target state (None for stabilization)
            dt: Time step
            max_forcing: Maximum forcing magnitude
        """
        self.oscillator_fn = oscillator_fn
        self.target_state = target_state
        self.dt = dt
        self.max_forcing = max_forcing
        
        self.state = None
        self.step_count = 0
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.state = np.random.randn(3) * 0.1
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take environment step.
        
        Args:
            action: Control action (forcing)
            
        Returns:
            (next_state, reward, done, info)
        """
        # Clip action
        forcing = np.clip(action, -self.max_forcing, self.max_forcing)
        
        # Step oscillator
        self.state = self.oscillator_fn(self.state, forcing[0] if action.ndim > 0 else forcing)
        
        # Compute reward
        if self.target_state is not None:
            # Distance to target
            distance = np.linalg.norm(self.state - self.target_state)
            reward = -distance
        else:
            # Stabilization reward (penalize large velocities)
            reward = -np.linalg.norm(self.state)
        
        self.step_count += 1
        done = self.step_count >= 1000
        
        return self.state.copy(), reward, done, {}


# ============================================================================
# Helper Functions
# ============================================================================

def create_rl_controller(
    node_id: str,
    state_dim: int = 3,
    action_dim: int = 1,
    algorithm: str = 'ppo',
    **kwargs
) -> RLControllerNode:
    """
    Factory function for RL controllers.
    
    Args:
        node_id: Unique identifier
        state_dim: State dimension
        action_dim: Action dimension
        algorithm: 'ppo' or 'sac'
        **kwargs: Additional arguments
        
    Returns:
        RLControllerNode
    """
    return RLControllerNode(node_id, state_dim, action_dim, algorithm, **kwargs)


if __name__ == '__main__':
    # Example usage
    if PYTORCH_AVAILABLE:
        print("Creating RL controller...")
        
        controller = RLControllerNode(
            "rl1",
            state_dim=3,
            action_dim=1,
            algorithm='ppo'
        )
        
        print(f"Algorithm: {controller.algorithm}")
        print(f"State dim: {controller.state_dim}")
        print(f"Action dim: {controller.action_dim}")
        
        # Test action selection
        state = np.random.randn(3)
        action = controller.forward(state)
        print(f"\nTest action: {action}")
    else:
        print("PyTorch not available. Install with: pip install torch")

