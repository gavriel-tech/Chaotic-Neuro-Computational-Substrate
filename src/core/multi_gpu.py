"""
Multi-GPU Support for GMCS.

Implements data parallelism using JAX's pmap for distributing
simulation across multiple GPUs.
"""

from typing import Tuple, List, Optional, Dict
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


# ============================================================================
# Multi-GPU THRML Sampling
# ============================================================================

class MultiGPUTHRMLSampler:
    """
    Multi-GPU THRML sampler using JAX pmap for parallel chain sampling.
    
    This class enables efficient sampling of multiple THRML chains across
    multiple GPUs, providing near-linear speedup for independent sampling.
    
    Key features:
    - Parallel chain sampling using pmap
    - Automatic device detection and allocation
    - Load balancing across GPUs
    - Gradient aggregation for CD learning
    - Compatible with all blocking strategies
    """
    
    def __init__(self, thrml_wrapper, n_devices: Optional[int] = None):
        """
        Initialize multi-GPU THRML sampler.
        
        Args:
            thrml_wrapper: THRMLWrapper instance to replicate
            n_devices: Number of devices to use (None = all available)
        """
        self.base_wrapper = thrml_wrapper
        self.devices = detect_devices()
        
        if n_devices is None:
            self.n_devices = len(self.devices)
        else:
            self.n_devices = min(n_devices, len(self.devices))
        
        self.devices = self.devices[:self.n_devices]
        
        print(f"[MultiGPU-THRML] Initialized with {self.n_devices} devices")
        for i, device in enumerate(self.devices):
            print(f"  Device {i}: {device}")
    
    def sample_parallel_chains(
        self,
        n_chains: int,
        n_steps: int,
        temperature: float,
        key: jax.random.PRNGKey
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Sample multiple THRML chains in parallel across GPUs.
        
        Args:
            n_chains: Total number of chains (distributed across GPUs)
            n_steps: Gibbs steps per chain
            temperature: Sampling temperature
            key: JAX random key
            
        Returns:
            Tuple of (samples, diagnostics)
            - samples: (n_chains, n_nodes) array of binary states
            - diagnostics: List of per-chain diagnostic dicts
        """
        # Ensure n_chains is divisible by n_devices
        chains_per_device = (n_chains + self.n_devices - 1) // self.n_devices
        total_chains = chains_per_device * self.n_devices
        
        print(f"[MultiGPU-THRML] Sampling {total_chains} chains ({chains_per_device} per device)")
        
        # Split keys for each chain
        keys = jax.random.split(key, total_chains)
        keys = keys.reshape(self.n_devices, chains_per_device, -1)
        
        # Define single-chain sampling function
        def sample_single_chain(chain_key):
            """Sample one chain (will be vmapped and pmapped)."""
            from thrml import Block, SamplingSchedule, sample_states
            from thrml.models import IsingSamplingProgram, hinton_init, IsingEBM
            
            # Update beta
            beta_temp = 1.0 / temperature
            beta_jax = jnp.array(beta_temp, dtype=jnp.float32)
            
            # Create model (using base wrapper's parameters)
            model = IsingEBM(
                nodes=self.base_wrapper.nodes,
                edges=self.base_wrapper.edges,
                biases=self.base_wrapper.biases_jax,
                weights=self.base_wrapper.edge_weights_jax,
                beta=beta_jax
            )
            
            # Create program
            program = IsingSamplingProgram(
                model=model,
                free_blocks=self.base_wrapper.free_blocks,
                clamped_blocks=[]
            )
            
            # Initialize
            k_init, k_samp = jax.random.split(chain_key)
            init_state = hinton_init(k_init, model, self.base_wrapper.free_blocks, ())
            
            # Sample
            schedule = SamplingSchedule(
                n_warmup=max(1, n_steps // 2),
                n_samples=1,
                steps_per_sample=2
            )
            
            samples = sample_states(
                k_samp,
                program,
                schedule,
                init_state,
                [],
                [Block(self.base_wrapper.nodes)]
            )
            
            return samples[-1]  # Return final sample
        
        # Vmap over chains per device
        vmapped_sample = jax.vmap(sample_single_chain)
        
        # Pmap across devices
        pmapped_sample = jax.pmap(vmapped_sample, devices=self.devices)
        
        # Execute
        import time
        t_start = time.time()
        
        all_samples = pmapped_sample(keys)
        
        # Wait for completion
        all_samples = jax.device_get(all_samples)
        
        wall_time = time.time() - t_start
        
        # Reshape from (n_devices, chains_per_device, n_nodes) to (total_chains, n_nodes)
        all_samples = np.array(all_samples).reshape(-1, self.base_wrapper.n_nodes)
        
        # Trim to requested n_chains
        all_samples = all_samples[:n_chains]
        
        # Compute diagnostics
        diagnostics = []
        for i in range(n_chains):
            sample = all_samples[i]
            energy = self.base_wrapper.compute_energy(sample)
            magnetization = float(np.mean(sample))
            
            diagnostics.append({
                'chain_id': i,
                'device_id': i % self.n_devices,
                'energy': energy,
                'magnetization': magnetization,
                'wall_time': wall_time
            })
        
        print(f"[MultiGPU-THRML] Completed {n_chains} chains in {wall_time:.3f}s "
              f"({n_chains/wall_time:.1f} chains/sec)")
        
        return all_samples, diagnostics
    
    def parallel_cd_update(
        self,
        data_states: np.ndarray,
        eta: float,
        k_steps: int,
        n_chains: int,
        key: jax.random.PRNGKey
    ) -> Dict[str, float]:
        """
        Parallel Contrastive Divergence update across multiple GPUs.
        
        Samples multiple chains in parallel, then aggregates gradients
        for a more accurate CD update.
        
        Args:
            data_states: (n_nodes,) observed data states
            eta: Learning rate
            k_steps: CD-k steps
            n_chains: Number of parallel chains
            key: JAX random key
            
        Returns:
            Diagnostic dict with gradient norms, energy differences
        """
        print(f"[MultiGPU-THRML] Running parallel CD-{k_steps} with {n_chains} chains")
        
        # Compute data statistics
        data_states_jax = jnp.array(data_states[:self.base_wrapper.n_nodes], dtype=jnp.float32)
        C_data = jnp.outer(data_states_jax, data_states_jax)
        data_energy = self.base_wrapper.compute_energy(data_states)
        
        # Sample multiple chains in parallel
        temperature = 1.0 / float(self.base_wrapper.beta)
        model_samples, _ = self.sample_parallel_chains(
            n_chains=n_chains,
            n_steps=k_steps,
            temperature=temperature,
            key=key
        )
        
        # Compute model statistics (averaged over chains)
        C_model = jnp.zeros_like(C_data)
        for sample in model_samples:
            sample_jax = jnp.array(sample, dtype=jnp.float32)
            C_model += jnp.outer(sample_jax, sample_jax)
        C_model /= n_chains
        
        # Compute gradient
        delta_W = eta * (C_data - C_model)
        gradient_norm = float(jnp.linalg.norm(delta_W))
        
        # Update weights in base wrapper
        self.base_wrapper._full_weights[:self.base_wrapper.n_nodes, :self.base_wrapper.n_nodes] += np.array(delta_W)
        
        # Ensure symmetry
        weights_upper = np.triu(self.base_wrapper._full_weights[:self.base_wrapper.n_nodes, :self.base_wrapper.n_nodes], k=1)
        self.base_wrapper._full_weights[:self.base_wrapper.n_nodes, :self.base_wrapper.n_nodes] = (
            weights_upper + weights_upper.T
        )
        np.fill_diagonal(self.base_wrapper._full_weights[:self.base_wrapper.n_nodes, :self.base_wrapper.n_nodes], 0.0)
        
        # Rebuild edge list
        self.base_wrapper.edges = []
        self.base_wrapper.edge_weights = []
        
        for i in range(self.base_wrapper.n_nodes):
            for j in range(i + 1, self.base_wrapper.n_nodes):
                w = self.base_wrapper._full_weights[i, j]
                if abs(w) > 1e-6:
                    self.base_wrapper.edges.append((self.base_wrapper.nodes[i], self.base_wrapper.nodes[j]))
                    self.base_wrapper.edge_weights.append(float(w))
        
        # Update JAX arrays
        self.base_wrapper.edge_weights_jax = jnp.array(
            self.base_wrapper.edge_weights if self.base_wrapper.edge_weights else [0.0],
            dtype=jnp.float32
        )
        
        # Recreate model
        from thrml.models import IsingEBM
        self.base_wrapper.model = IsingEBM(
            nodes=self.base_wrapper.nodes,
            edges=self.base_wrapper.edges,
            biases=self.base_wrapper.biases_jax,
            weights=self.base_wrapper.edge_weights_jax,
            beta=self.base_wrapper.beta_jax
        )
        
        # Compute model energy (use mean of samples)
        model_energy = np.mean([self.base_wrapper.compute_energy(s) for s in model_samples])
        
        return {
            'gradient_norm': gradient_norm,
            'data_energy': data_energy,
            'model_energy': float(model_energy),
            'energy_diff': data_energy - float(model_energy),
            'n_chains': n_chains,
            'n_devices': self.n_devices,
            'k_steps': k_steps
        }


def create_multi_gpu_thrml_sampler(
    thrml_wrapper,
    n_devices: Optional[int] = None
) -> Optional[MultiGPUTHRMLSampler]:
    """
    Factory function to create multi-GPU THRML sampler.
    
    Args:
        thrml_wrapper: THRMLWrapper instance
        n_devices: Number of devices (None = all)
        
    Returns:
        MultiGPUTHRMLSampler or None if only 1 device available
    """
    devices = detect_devices()
    
    if len(devices) < 2:
        print("[MultiGPU-THRML] Only 1 device available, multi-GPU disabled")
        return None
    
    try:
        return MultiGPUTHRMLSampler(thrml_wrapper, n_devices)
    except Exception as e:
        print(f"[MultiGPU-THRML] Failed to create multi-GPU sampler: {e}")
        return None
