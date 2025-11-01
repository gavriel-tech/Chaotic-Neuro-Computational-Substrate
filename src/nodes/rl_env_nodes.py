"""
Reinforcement Learning Environment Nodes for GMCS Node Graph.

Wraps OpenAI Gym environments for use in node graphs.
Supports standard RL loop: reset, step, render.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Optional Gym dependency
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
    print("[INFO] Using gymnasium")
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
        print("[INFO] Using gym (legacy)")
    except ImportError:
        GYM_AVAILABLE = False
        print("[WARNING] gym/gymnasium not available - RL environments will not work")


# ============================================================================
# RL Environment Node
# ============================================================================

@dataclass
class RLEnvironmentConfig:
    """Configuration for RL environment."""
    env_name: str = "CartPole-v1"
    render: bool = False
    max_episode_steps: int = 500
    auto_reset: bool = True


class RLEnvironmentNode:
    """
    Wrap an OpenAI Gym environment as a node.
    
    Provides standard RL interface: state, action, reward, done.
    Automatically resets when episode ends.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'rl_env')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        self.config = RLEnvironmentConfig(**config)
        self.env = None
        self.current_state = None
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0
        self.total_steps = 0
        self.done = False
        
        if GYM_AVAILABLE:
            self._initialize_environment()
        else:
            print("[RLEnvironmentNode] Gym not available - using stub")
            self._initialize_stub()
    
    def _initialize_environment(self):
        """Initialize the Gym environment."""
        try:
            render_mode = "human" if self.config.render else None
            self.env = gym.make(self.config.env_name, render_mode=render_mode)
            
            # Reset to get initial state
            if hasattr(self.env, 'reset'):
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    self.current_state, _ = reset_result
                else:
                    self.current_state = reset_result
            
            print(f"[RLEnvironmentNode] Initialized environment: {self.config.env_name}")
            print(f"  State space: {self.env.observation_space}")
            print(f"  Action space: {self.env.action_space}")
            
        except Exception as e:
            print(f"[RLEnvironmentNode ERROR] Failed to create environment: {e}")
            self._initialize_stub()
    
    def _initialize_stub(self):
        """Initialize a stub environment for testing without Gym."""
        self.current_state = np.array([0.0, 0.0, 0.0, 0.0])  # CartPole-like state
        self.state_dim = 4
        self.action_dim = 2
        print("[RLEnvironmentNode] Using stub environment")
    
    def process(self, action: Optional[Any] = None, reset: bool = False, **inputs) -> Dict[str, Any]:
        """
        Step the environment or reset.
        
        Args:
            action: Action to take (if None, takes random action)
            reset: Force reset the environment
            **inputs: Additional inputs
            
        Returns:
            Dictionary with state, reward, done, info, render
        """
        if reset or (self.done and self.config.auto_reset):
            return self._reset()
        
        # Get action
        if action is None:
            # Take random action if none provided
            if self.env:
                action = self.env.action_space.sample()
            else:
                action = np.random.randint(0, 2)  # Stub: binary action
        
        # Convert action to proper format
        action = self._process_action(action)
        
        # Step environment
        if self.env:
            step_result = self.env.step(action)
            
            # Handle both gym and gymnasium return formats
            if len(step_result) == 5:
                # Gymnasium format: (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Legacy gym format: (obs, reward, done, info)
                next_state, reward, done, info = step_result
            
            self.current_state = next_state
            self.done = done
        else:
            # Stub environment
            next_state, reward, done, info = self._stub_step(action)
            self.current_state = next_state
            self.done = done
        
        # Update metrics
        self.episode_reward += reward
        self.episode_length += 1
        self.total_steps += 1
        
        # Auto-reset if done
        if done:
            print(f"[RLEnvironmentNode] Episode {self.episode_count} finished. Reward: {self.episode_reward:.2f}, Length: {self.episode_length}")
            episode_reward = self.episode_reward
            episode_length = self.episode_length
            
            if self.config.auto_reset:
                reset_result = self._reset()
                reset_result['episode_reward'] = episode_reward
                reset_result['episode_length'] = episode_length
                return reset_result
        
        return {
            'state': self.current_state.copy() if isinstance(self.current_state, np.ndarray) else np.array(self.current_state),
            'reward': float(reward),
            'done': bool(done),
            'info': info if self.env else {},
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'render': self._get_render() if self.config.render else None,
            'experience': {
                'state': self.current_state.copy() if isinstance(self.current_state, np.ndarray) else np.array(self.current_state),
                'action': action,
                'reward': float(reward),
                'next_state': next_state.copy() if isinstance(next_state, np.ndarray) else np.array(next_state),
                'done': bool(done)
            }
        }
    
    def _reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        if self.env:
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                self.current_state, info = reset_result
            else:
                self.current_state = reset_result
                info = {}
        else:
            # Stub reset
            self.current_state = np.random.randn(4) * 0.1
            info = {}
        
        self.episode_count += 1
        self.episode_reward = 0.0
        self.episode_length = 0
        self.done = False
        
        return {
            'state': self.current_state.copy() if isinstance(self.current_state, np.ndarray) else np.array(self.current_state),
            'reward': 0.0,
            'done': False,
            'info': info,
            'episode_reward': 0.0,
            'episode_length': 0,
            'render': None
        }
    
    def _process_action(self, action: Any) -> Any:
        """Convert action to environment format."""
        if self.env is None:
            return action
        
        # Handle continuous vs discrete action spaces
        if hasattr(self.env.action_space, 'n'):
            # Discrete action space
            if isinstance(action, np.ndarray):
                action = int(action.flatten()[0])
            else:
                action = int(action)
        else:
            # Continuous action space
            if not isinstance(action, np.ndarray):
                action = np.array([action])
        
        return action
    
    def _stub_step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Stub environment step (CartPole-like dynamics).
        """
        # Simple physics simulation
        x, x_dot, theta, theta_dot = self.current_state
        
        force = 1.0 if action == 1 else -1.0
        
        # Simplified CartPole dynamics
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        length = 0.5
        
        temp = (force + masspole * length * theta_dot**2 * np.sin(theta)) / (masscart + masspole)
        thetaacc = (gravity * np.sin(theta) - np.cos(theta) * temp) / (length * (4.0/3.0 - masspole * np.cos(theta)**2 / (masscart + masspole)))
        xacc = temp - masspole * length * thetaacc * np.cos(theta) / (masscart + masspole)
        
        dt = 0.02
        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc
        
        next_state = np.array([x, x_dot, theta, theta_dot])
        
        # Check done conditions
        done = bool(
            x < -2.4 or x > 2.4 or
            theta < -0.2095 or theta > 0.2095 or
            self.episode_length >= self.config.max_episode_steps
        )
        
        reward = 1.0 if not done else 0.0
        
        return next_state, reward, done, {}
    
    def _get_render(self) -> Optional[np.ndarray]:
        """Get render frame if available."""
        if self.env and hasattr(self.env, 'render'):
            try:
                frame = self.env.render()
                return frame
            except:
                return None
        return None
    
    def close(self):
        """Close the environment."""
        if self.env:
            self.env.close()
            print("[RLEnvironmentNode] Environment closed")
    
    def __del__(self):
        self.close()


# ============================================================================
# Node Factory Registration Helper
# ============================================================================

def create_rl_env_node(node_name: str, config: Dict[str, Any]) -> RLEnvironmentNode:
    """
    Factory function to create RL environment nodes.
    
    Args:
        node_name: Name of the environment node
        config: Node configuration
        
    Returns:
        RLEnvironmentNode instance
    """
    return RLEnvironmentNode(config)


# ============================================================================
# Alias for Test Compatibility
# ============================================================================

# Tests expect this exact class name
GymEnvironmentNode = RLEnvironmentNode

