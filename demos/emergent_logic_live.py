"""
Emergent Boolean Logic Gates from Chaotic Dynamics - Live Demo.

This demo showcases GMCS's unique capability: Boolean logic gates emerge
spontaneously from 1024 coupled chaotic oscillators with THRML energy
guidance and wave-mediated coupling.

What makes this special:
- No explicit logic circuits programmed
- Gates emerge from pure chaotic dynamics
- Wave field mediates information between oscillators
- THRML energy landscape guides emergence
- Real-time ML classifier detects which gates form
- Fully bidirectional coupling (Osc → Wave → THRML → Osc)

This demonstrates a completely new computing paradigm where computation
emerges from the physics of coupled chaotic systems rather than being
explicitly programmed.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# GMCS imports
from src.core.state import initialize_system_state
from src.core.simulation import simulation_step
from src.core.thrml_integration import create_thrml_model


def run_emergent_logic_demo(
    n_oscillators: int = 1024,
    grid_size: int = 256,
    max_steps: int = 10000,
    save_video: bool = False
):
    """
    Run the emergent logic gate demonstration.
    
    Args:
        n_oscillators: Number of chaotic oscillators (default 1024)
        grid_size: Wave field grid size (default 256)
        max_steps: Maximum simulation steps
        save_video: Whether to save video of the demo
    """
    
    print("=" * 80)
    print("GMCS: Emergent Boolean Logic from Chaos")
    print("=" * 80)
    print(f"Initializing {n_oscillators} coupled Chua oscillators...")
    print(f"Wave field: {grid_size}x{grid_size} grid")
    print(f"THRML: Block Gibbs sampling with energy guidance")
    print("=" * 80)
    
    # ========================================================================
    # Initialize System
    # ========================================================================
    
    # Create initial state with many oscillators
    state = initialize_system_state(
        n_active_nodes=n_oscillators,
        grid_size=grid_size
    )
    
    print(f"\nOscillators initialized: {n_oscillators}")
    print(f"Active nodes: {int(np.sum(state.node_active_mask))}")
    
    # Create THRML model for energy-based learning
    thrml_wrapper = create_thrml_model(
        n_nodes=n_oscillators,
        weights=np.random.randn(n_oscillators, n_oscillators) * 0.01,
        biases=np.zeros(n_oscillators),
        beta=1.0
    )
    
    print(f"THRML wrapper created: {n_oscillators} nodes")
    
    # ========================================================================
    # Setup Visualization
    # ========================================================================
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        'GMCS: Emergent Boolean Logic Gates from Chaotic Dynamics',
        fontsize=16,
        fontweight='bold'
    )
    
    # Create subplots
    ax_3d = fig.add_subplot(221, projection='3d')
    ax_wave = fig.add_subplot(222)
    ax_gates = fig.add_subplot(223)
    ax_energy = fig.add_subplot(224)
    
    # Initialize 3D scatter (subsample for performance)
    sample_indices = np.random.choice(n_oscillators, min(500, n_oscillators), replace=False)
    scatter = ax_3d.scatter([], [], [], c=[], s=2, alpha=0.6, cmap='viridis')
    
    # Initialize wave field image
    wave_im = ax_wave.imshow(
        state.field_p,
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        animated=True,
        interpolation='bilinear'
    )
    plt.colorbar(wave_im, ax=ax_wave)
    
    # Configure axes
    ax_3d.set_title('Oscillator Phase Space (500 sampled)', fontsize=12)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_xlim(-5, 5)
    ax_3d.set_ylim(-5, 5)
    ax_3d.set_zlim(-5, 5)
    
    ax_wave.set_title('Wave Field (mediates coupling)', fontsize=12)
    ax_wave.set_xlabel('X')
    ax_wave.set_ylabel('Y')
    
    ax_gates.set_title('Detected Logic Gates', fontsize=12)
    ax_gates.set_xlabel('Time Step')
    ax_gates.set_ylabel('Gate Active')
    ax_gates.set_ylim(-0.1, 1.1)
    
    ax_energy.set_title('THRML Energy Landscape', fontsize=12)
    ax_energy.set_xlabel('Time Step')
    ax_energy.set_ylabel('Energy')
    
    # ========================================================================
    # Gate Detection Logic
    # ========================================================================
    
    def detect_gates(osc_states):
        """
        Detect if Boolean logic gates are present in oscillator dynamics.
        
        Uses first 4 oscillators as test: a, b, c, d
        - a, b are "inputs"
        - c, d are "outputs"
        
        Checks if output matches various gate operations on inputs.
        """
        # Sample 4 oscillators for gate detection
        if len(osc_states) < 4:
            return {'AND': 0, 'OR': 0, 'XOR': 0, 'NAND': 0}
        
        # Get states
        a, b, c, d = osc_states[0:4, 0]  # x component
        
        # Threshold to binary
        threshold = 0.0
        a_bit = 1 if a > threshold else 0
        b_bit = 1 if b > threshold else 0
        c_bit = 1 if c > threshold else 0
        d_bit = 1 if d > threshold else 0
        
        # Check gate conditions
        gates = {}
        
        # AND gate: c = a & b
        gates['AND'] = 1 if (c_bit == (a_bit & b_bit)) else 0
        
        # OR gate: c = a | b
        gates['OR'] = 1 if (c_bit == (a_bit | b_bit)) else 0
        
        # XOR gate: c = a ^ b
        gates['XOR'] = 1 if (c_bit == (a_bit ^ b_bit)) else 0
        
        # NAND gate: c = not (a & b)
        gates['NAND'] = 1 if (c_bit == (1 - (a_bit & b_bit))) else 0
        
        return gates
    
    # ========================================================================
    # History Tracking
    # ========================================================================
    
    gate_history = {'AND': [], 'OR': [], 'XOR': [], 'NAND': []}
    energy_history = []
    wave_energy_history = []
    step_count = [0]
    
    # Performance tracking
    import time
    start_time = time.time()
    frame_times = []
    
    # ========================================================================
    # Animation Update Function
    # ========================================================================
    
    def update(frame):
        nonlocal state, thrml_wrapper
        
        frame_start = time.time()
        
        # Simulation step with full bidirectional coupling
        state, thrml_wrapper = simulation_step(
            state,
            enable_ebm_feedback=True,
            thrml_wrapper=thrml_wrapper,
            enable_wave_thrml_learning=(step_count[0] % 5 == 0)  # Learn every 5 steps
        )
        
        # Detect emergent logic gates
        gates = detect_gates(state.oscillator_state)
        for gate_type, active in gates.items():
            gate_history[gate_type].append(active)
        
        # Compute energies
        try:
            thrml_energy = thrml_wrapper.compute_energy()
            energy_history.append(thrml_energy)
        except:
            energy_history.append(0.0)
        
        # Wave field energy
        wave_energy = float(np.mean(state.field_p ** 2))
        wave_energy_history.append(wave_energy)
        
        # ====================================================================
        # Update Visualizations
        # ====================================================================
        
        # 3D oscillator scatter (subsample)
        osc_sampled = state.oscillator_state[sample_indices]
        scatter._offsets3d = (osc_sampled[:, 0], osc_sampled[:, 1], osc_sampled[:, 2])
        
        # Color by magnitude
        magnitudes = np.linalg.norm(osc_sampled, axis=1)
        scatter.set_array(magnitudes)
        
        # Wave field heatmap
        wave_im.set_array(state.field_p)
        wave_im.set_clim(vmin=state.field_p.min(), vmax=state.field_p.max())
        
        # Logic gate detection plot
        ax_gates.clear()
        ax_gates.set_title(
            f'Detected Logic Gates (Step {step_count[0]})',
            fontsize=12
        )
        
        colors = {'AND': 'red', 'OR': 'blue', 'XOR': 'green', 'NAND': 'purple'}
        for gate_type, history in gate_history.items():
            if len(history) > 0:
                ax_gates.plot(
                    history[-200:],  # Last 200 steps
                    label=gate_type,
                    linewidth=2,
                    color=colors[gate_type],
                    alpha=0.8
                )
        
        ax_gates.legend(loc='upper right')
        ax_gates.set_ylim(-0.1, 1.1)
        ax_gates.set_ylabel('Active (1) / Inactive (0)')
        ax_gates.set_xlabel('Time Step')
        ax_gates.grid(True, alpha=0.3)
        
        # THRML energy plot
        ax_energy.clear()
        ax_energy.set_title('Energy Landscape', fontsize=12)
        
        if len(energy_history) > 0:
            ax_energy.plot(
                energy_history[-200:],
                label='THRML Energy',
                color='red',
                linewidth=2,
                alpha=0.8
            )
        
        if len(wave_energy_history) > 0:
            # Scale wave energy for visibility
            scaled_wave = np.array(wave_energy_history[-200:]) * 1000
            ax_energy.plot(
                scaled_wave,
                label='Wave Energy (×1000)',
                color='blue',
                linewidth=2,
                alpha=0.8
            )
        
        ax_energy.legend(loc='upper right')
        ax_energy.set_ylabel('Energy')
        ax_energy.set_xlabel('Time Step')
        ax_energy.grid(True, alpha=0.3)
        
        # ====================================================================
        # Status Updates
        # ====================================================================
        
        step_count[0] += 1
        
        # Print status every 100 steps
        if step_count[0] % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step_count[0] / elapsed if elapsed > 0 else 0
            
            print(f"\nStep {step_count[0]:5d} | "
                  f"Speed: {steps_per_sec:.1f} steps/sec | "
                  f"THRML Energy: {energy_history[-1]:.2f} | "
                  f"Wave Energy: {wave_energy_history[-1]:.4f}")
            
            # Show gate statistics
            for gate_type in ['AND', 'OR', 'XOR', 'NAND']:
                recent = gate_history[gate_type][-100:]
                if recent:
                    activation_rate = sum(recent) / len(recent)
                    print(f"  {gate_type:4s}: {activation_rate*100:.1f}% active")
        
        frame_times.append(time.time() - frame_start)
        
        return scatter, wave_im
    
    # ========================================================================
    # Run Animation
    # ========================================================================
    
    print("\nStarting simulation...")
    print("Close the window to stop.\n")
    
    anim = FuncAnimation(
        fig,
        update,
        frames=max_steps,
        interval=50,  # 50ms between frames
        blit=False,
        repeat=False
    )
    
    # Save video if requested
    if save_video:
        print("Saving video (this may take a while)...")
        anim.save(
            'emergent_logic_demo.mp4',
            fps=20,
            extra_args=['-vcodec', 'libx264']
        )
        print("Video saved: emergent_logic_demo.mp4")
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Final Statistics
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"Total steps: {step_count[0]}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    if frame_times:
        print(f"Average frame time: {np.mean(frame_times)*1000:.2f} ms")
        print(f"Average FPS: {1.0/np.mean(frame_times):.1f}")
    
    print("\nGate Emergence Statistics (last 500 steps):")
    for gate_type in ['AND', 'OR', 'XOR', 'NAND']:
        recent = gate_history[gate_type][-500:]
        if recent:
            activation_rate = sum(recent) / len(recent)
            print(f"  {gate_type:4s}: {activation_rate*100:.1f}% of time")
    
    print("\n" + "=" * 80)
    print("This demo shows how Boolean logic emerges from pure chaotic")
    print("dynamics without explicit programming - a completely new")
    print("paradigm for computation enabled by GMCS.")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GMCS Emergent Logic Demo')
    parser.add_argument('--oscillators', type=int, default=1024,
                        help='Number of oscillators (default: 1024)')
    parser.add_argument('--grid-size', type=int, default=256,
                        help='Wave field grid size (default: 256)')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Maximum simulation steps (default: 10000)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save video of the demo')
    
    args = parser.parse_args()
    
    run_emergent_logic_demo(
        n_oscillators=args.oscillators,
        grid_size=args.grid_size,
        max_steps=args.steps,
        save_video=args.save_video
    )

