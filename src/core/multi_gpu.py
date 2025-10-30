"""
Multi-GPU Support for GMCS.

Implements data parallelism using JAX's pmap for distributing
simulation across multiple GPUs.
"""

from typing import Tuple, List, Optional
import jax
import jax.numpy as jnp
from jax import pmap, device_put
import numpy as np

from .state import SystemState, N_MAX


def detect_devices() -> List[jax.Device]:
    """
    Detect available JAX devices (GPUs, TPUs, CPUs).
    
    Returns:
        List of available devices
    """
    devices = jax.devices()
    return devices


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    devices = detect_devices()
    
    info = {
        'n_devices': len(devices),
        'devices': [],
        'default_backend': jax.default_backend()
    }
    
    for i, device in enumerate(devices):
        device_info = {
            'id': i,
            'platform': device.platform,
            'device_kind': device.device_kind,
        }
        info['devices'].append(device_info)
    
    return info


def shard_state_for_devices(
    state: SystemState,
    n_devices: int
) -> Tuple[SystemState, ...]:
    """
    Shard system state across devices.
    
    Splits nodes across devices for parallel processing.
    Each device gets N_MAX/n_devices nodes.
    
    Args:
        state: SystemState to shard
        n_devices: Number of devices
        
    Returns:
        Tuple of SystemState, one per device
    """
    nodes_per_device = N_MAX // n_devices
    
    sharded_states = []
    for device_id in range(n_devices):
        start_idx = device_id * nodes_per_device
        end_idx = start_idx + nodes_per_device
        
        # Create view of state for this device's nodes
        # Note: This is simplified - full implementation would need
        # proper sharding of all arrays
        device_state = state._replace(
            oscillator_state=state.oscillator_state[start_idx:end_idx],
            node_active_mask=state.node_active_mask[start_idx:end_idx],
            node_positions=state.node_positions[start_idx:end_idx],
            # GMCS parameters
            gmcs_A_max=state.gmcs_A_max[start_idx:end_idx],
            gmcs_R_comp=state.gmcs_R_comp[start_idx:end_idx],
            gmcs_T_comp=state.gmcs_T_comp[start_idx:end_idx],
            gmcs_R_exp=state.gmcs_R_exp[start_idx:end_idx],
            gmcs_T_exp=state.gmcs_T_exp[start_idx:end_idx],
            gmcs_Phi=state.gmcs_Phi[start_idx:end_idx],
            gmcs_omega=state.gmcs_omega[start_idx:end_idx],
            gmcs_gamma=state.gmcs_gamma[start_idx:end_idx],
            gmcs_beta=state.gmcs_beta[start_idx:end_idx],
            gmcs_f0=state.gmcs_f0[start_idx:end_idx],
            gmcs_Q=state.gmcs_Q[start_idx:end_idx],
            gmcs_levels=state.gmcs_levels[start_idx:end_idx],
            gmcs_rate_limit=state.gmcs_rate_limit[start_idx:end_idx],
            gmcs_n2=state.gmcs_n2[start_idx:end_idx],
            gmcs_V=state.gmcs_V[start_idx:end_idx],
            gmcs_V_pi=state.gmcs_V_pi[start_idx:end_idx],
            k_strengths=state.k_strengths[start_idx:end_idx],
        )
        
        sharded_states.append(device_state)
    
    return tuple(sharded_states)


def merge_sharded_states(
    sharded_states: Tuple[SystemState, ...],
    original_state: SystemState
) -> SystemState:
    """
    Merge sharded states back into single state.
    
    Args:
        sharded_states: Tuple of states from each device
        original_state: Original state (for non-sharded fields)
        
    Returns:
        Merged SystemState
    """
    # Concatenate node-specific arrays
    oscillator_state = jnp.concatenate([
        s.oscillator_state for s in sharded_states
    ], axis=0)
    
    node_active_mask = jnp.concatenate([
        s.node_active_mask for s in sharded_states
    ], axis=0)
    
    node_positions = jnp.concatenate([
        s.node_positions for s in sharded_states
    ], axis=0)
    
    # Merge GMCS parameters
    gmcs_A_max = jnp.concatenate([s.gmcs_A_max for s in sharded_states])
    gmcs_R_comp = jnp.concatenate([s.gmcs_R_comp for s in sharded_states])
    gmcs_T_comp = jnp.concatenate([s.gmcs_T_comp for s in sharded_states])
    gmcs_R_exp = jnp.concatenate([s.gmcs_R_exp for s in sharded_states])
    gmcs_T_exp = jnp.concatenate([s.gmcs_T_exp for s in sharded_states])
    gmcs_Phi = jnp.concatenate([s.gmcs_Phi for s in sharded_states])
    gmcs_omega = jnp.concatenate([s.gmcs_omega for s in sharded_states])
    gmcs_gamma = jnp.concatenate([s.gmcs_gamma for s in sharded_states])
    gmcs_beta = jnp.concatenate([s.gmcs_beta for s in sharded_states])
    gmcs_f0 = jnp.concatenate([s.gmcs_f0 for s in sharded_states])
    gmcs_Q = jnp.concatenate([s.gmcs_Q for s in sharded_states])
    gmcs_levels = jnp.concatenate([s.gmcs_levels for s in sharded_states])
    gmcs_rate_limit = jnp.concatenate([s.gmcs_rate_limit for s in sharded_states])
    gmcs_n2 = jnp.concatenate([s.gmcs_n2 for s in sharded_states])
    gmcs_V = jnp.concatenate([s.gmcs_V for s in sharded_states])
    gmcs_V_pi = jnp.concatenate([s.gmcs_V_pi for s in sharded_states])
    k_strengths = jnp.concatenate([s.k_strengths for s in sharded_states])
    
    # Use wave field from first device (shared resource)
    # In full implementation, would need proper reduction
    field_p = sharded_states[0].field_p
    field_p_prev = sharded_states[0].field_p_prev
    
    # Create merged state
    merged_state = original_state._replace(
        oscillator_state=oscillator_state,
        node_active_mask=node_active_mask,
        node_positions=node_positions,
        field_p=field_p,
        field_p_prev=field_p_prev,
        gmcs_A_max=gmcs_A_max,
        gmcs_R_comp=gmcs_R_comp,
        gmcs_T_comp=gmcs_T_comp,
        gmcs_R_exp=gmcs_R_exp,
        gmcs_T_exp=gmcs_T_exp,
        gmcs_Phi=gmcs_Phi,
        gmcs_omega=gmcs_omega,
        gmcs_gamma=gmcs_gamma,
        gmcs_beta=gmcs_beta,
        gmcs_f0=gmcs_f0,
        gmcs_Q=gmcs_Q,
        gmcs_levels=gmcs_levels,
        gmcs_rate_limit=gmcs_rate_limit,
        gmcs_n2=gmcs_n2,
        gmcs_V=gmcs_V,
        gmcs_V_pi=gmcs_V_pi,
        k_strengths=k_strengths,
    )
    
    return merged_state


class MultiGPUSimulator:
    """
    Multi-GPU simulator using JAX pmap.
    
    Distributes simulation across multiple GPUs for improved performance.
    """
    
    def __init__(self, n_devices: Optional[int] = None):
        """
        Initialize multi-GPU simulator.
        
        Args:
            n_devices: Number of devices to use (None = all available)
        """
        self.devices = detect_devices()
        
        if n_devices is None:
            self.n_devices = len(self.devices)
        else:
            self.n_devices = min(n_devices, len(self.devices))
        
        self.devices = self.devices[:self.n_devices]
        
        print(f"MultiGPUSimulator initialized with {self.n_devices} devices")
        for i, device in enumerate(self.devices):
            print(f"  Device {i}: {device.platform} - {device.device_kind}")
    
    def create_pmapped_step(self, step_fn):
        """
        Create pmapped version of simulation step function.
        
        Args:
            step_fn: Single-device simulation step function
            
        Returns:
            Pmapped function for multi-device execution
        """
        # Create pmapped version
        pmapped_fn = pmap(step_fn, devices=self.devices)
        return pmapped_fn
    
    def distribute_state(self, state: SystemState) -> SystemState:
        """
        Distribute state across devices.
        
        Args:
            state: SystemState to distribute
            
        Returns:
            State with arrays replicated across devices
        """
        # For pmap, we need to add a leading device dimension
        # This is a simplified version - full implementation would
        # properly shard the state
        
        # Replicate state across devices
        replicated_state = jax.tree_map(
            lambda x: jnp.stack([x] * self.n_devices),
            state
        )
        
        return replicated_state
    
    def gather_state(self, distributed_state: SystemState) -> SystemState:
        """
        Gather state from devices back to host.
        
        Args:
            distributed_state: State distributed across devices
            
        Returns:
            Gathered state on host
        """
        # Take first replica (they should all be the same for replicated computation)
        gathered_state = jax.tree_map(
            lambda x: x[0],
            distributed_state
        )
        
        return gathered_state


def benchmark_multi_gpu(
    n_devices: int,
    n_steps: int = 100
) -> dict:
    """
    Benchmark multi-GPU performance.
    
    Args:
        n_devices: Number of devices to use
        n_steps: Number of simulation steps
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    from .state import initialize_system_state
    from .simulation import simulation_step
    
    # Initialize state
    key = jax.random.PRNGKey(0)
    state = initialize_system_state(key)
    
    # Single GPU benchmark
    start = time.time()
    for _ in range(n_steps):
        state, _ = simulation_step(state, enable_ebm_feedback=False)
    single_gpu_time = time.time() - start
    
    # Multi-GPU benchmark
    if n_devices > 1:
        simulator = MultiGPUSimulator(n_devices=n_devices)
        distributed_state = simulator.distribute_state(state)
        
        # Create pmapped step
        pmapped_step = simulator.create_pmapped_step(
            lambda s: simulation_step(s, enable_ebm_feedback=False)[0]
        )
        
        start = time.time()
        for _ in range(n_steps):
            distributed_state = pmapped_step(distributed_state)
        multi_gpu_time = time.time() - start
        
        speedup = single_gpu_time / multi_gpu_time
    else:
        multi_gpu_time = single_gpu_time
        speedup = 1.0
    
    return {
        'n_devices': n_devices,
        'n_steps': n_steps,
        'single_gpu_time': single_gpu_time,
        'multi_gpu_time': multi_gpu_time,
        'speedup': speedup,
        'single_gpu_fps': n_steps / single_gpu_time,
        'multi_gpu_fps': n_steps / multi_gpu_time
    }

