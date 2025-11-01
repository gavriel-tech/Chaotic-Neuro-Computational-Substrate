"""
THRML-Specific Visualizers

Provides visualizations tailored for thermodynamic computing:
- P-bit (probabilistic bit) state grids
- Energy landscapes
- Autocorrelation plots
- Mixing curves (convergence)
- Blocking strategy visualization
- Chain diagnostics
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING
import logging

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, visualizations disabled")
    # Dummy types for type hints when matplotlib not available
    if TYPE_CHECKING:
        from matplotlib.figure import Figure
    else:
        Figure = Any

logger = logging.getLogger(__name__)


class THRMLVisualizer:
    """
    Comprehensive visualizer for THRML data.
    
    Features:
    - P-bit grid visualization
    - Energy landscape plots
    - Autocorrelation analysis
    - Mixing curve visualization
    - Block coloring display
    - Multi-chain comparison
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for THRML visualization")
        
        self.figsize = figsize
        self.fig = None
        self.axes = None
    
    def plot_pbit_grid(
        self,
        states: np.ndarray,
        grid_shape: Optional[Tuple[int, int]] = None,
        title: str = "P-bit State Grid",
        cmap: str = "RdBu",
        show_colorbar: bool = True
    ) -> Figure:
        """
        Visualize P-bit states as a 2D grid.
        
        Args:
            states: (n_nodes,) array of binary states {-1, +1}
            grid_shape: Optional (rows, cols) for grid layout
            title: Plot title
            cmap: Colormap name
            show_colorbar: Whether to show colorbar
            
        Returns:
            matplotlib Figure
        """
        # Auto-determine grid shape
        if grid_shape is None:
            n_nodes = len(states)
            side = int(np.ceil(np.sqrt(n_nodes)))
            grid_shape = (side, side)
        
        rows, cols = grid_shape
        
        # Pad states if needed
        n_needed = rows * cols
        if len(states) < n_needed:
            padded_states = np.zeros(n_needed)
            padded_states[:len(states)] = states
        else:
            padded_states = states[:n_needed]
        
        # Reshape to grid
        grid = padded_states.reshape(rows, cols)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot
        im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # Colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Spin State', fontsize=12)
            cbar.set_ticks([-1, 0, 1])
            cbar.set_ticklabels(['-1', '0', '+1'])
        
        # Grid lines
        ax.set_xticks(np.arange(cols) - 0.5, minor=True)
        ax.set_yticks(np.arange(rows) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_energy_landscape(
        self,
        samples: np.ndarray,
        energies: np.ndarray,
        title: str = "Energy Landscape",
        n_bins: int = 50
    ) -> Figure:
        """
        Plot energy distribution and trajectory.
        
        Args:
            samples: (n_samples, n_nodes) array of states
            energies: (n_samples,) array of energies
            title: Plot title
            n_bins: Number of histogram bins
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Energy trajectory
        axes[0].plot(energies, linewidth=1, alpha=0.7)
        axes[0].set_xlabel('Sample Index', fontsize=12)
        axes[0].set_ylabel('Energy', fontsize=12)
        axes[0].set_title(f'{title} - Trajectory', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add mean line
        mean_energy = np.mean(energies)
        axes[0].axhline(mean_energy, color='red', linestyle='--', 
                       label=f'Mean: {mean_energy:.2f}')
        axes[0].legend()
        
        # Energy histogram
        axes[1].hist(energies, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].set_xlabel('Energy', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title(f'{title} - Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        std_energy = np.std(energies)
        min_energy = np.min(energies)
        max_energy = np.max(energies)
        
        stats_text = f'μ={mean_energy:.2f}\nσ={std_energy:.2f}\nmin={min_energy:.2f}\nmax={max_energy:.2f}'
        axes[1].text(0.98, 0.98, stats_text,
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_autocorrelation(
        self,
        samples: np.ndarray,
        max_lag: int = 100,
        title: str = "Autocorrelation Analysis"
    ) -> Figure:
        """
        Plot autocorrelation function.
        
        Args:
            samples: (n_samples, n_nodes) array of states
            max_lag: Maximum lag to compute
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        n_samples, n_nodes = samples.shape
        max_lag = min(max_lag, n_samples - 1)
        
        # Compute autocorrelation for each node
        lags = np.arange(max_lag + 1)
        autocorrs = np.zeros((n_nodes, len(lags)))
        
        for i in range(n_nodes):
            node_samples = samples[:, i]
            mean = np.mean(node_samples)
            var = np.var(node_samples)
            
            if var > 1e-10:  # Avoid division by zero
                for lag in lags:
                    if lag == 0:
                        autocorrs[i, lag] = 1.0
                    else:
                        c = np.mean((node_samples[:-lag] - mean) * (node_samples[lag:] - mean))
                        autocorrs[i, lag] = c / var
        
        # Average across nodes
        mean_autocorr = np.mean(autocorrs, axis=0)
        std_autocorr = np.std(autocorrs, axis=0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(lags, mean_autocorr, linewidth=2, label='Mean')
        ax.fill_between(lags, 
                        mean_autocorr - std_autocorr,
                        mean_autocorr + std_autocorr,
                        alpha=0.3, label='±1 σ')
        
        # Reference lines
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(np.exp(-1), color='red', linestyle='--', linewidth=1,
                  label='e⁻¹ threshold')
        
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Estimate τ_int (integrated autocorrelation time)
        # Find where autocorr drops below e^-1
        threshold = np.exp(-1)
        tau_int_idx = np.where(mean_autocorr < threshold)[0]
        if len(tau_int_idx) > 0:
            tau_int = tau_int_idx[0]
            ax.axvline(tau_int, color='green', linestyle=':', linewidth=2,
                      label=f'τ_int ≈ {tau_int}')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_mixing_curves(
        self,
        samples: np.ndarray,
        energies: np.ndarray,
        window_size: int = 50,
        title: str = "Mixing Curves"
    ) -> Figure:
        """
        Plot mixing convergence curves.
        
        Args:
            samples: (n_samples, n_nodes) array of states
            energies: (n_samples,) array of energies
            window_size: Window for moving average
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        n_samples, n_nodes = samples.shape
        
        # Compute moving averages
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w
        
        # Energy moving average
        energy_ma = moving_average(energies, window_size)
        
        # Magnetization (mean spin)
        magnetization = np.mean(samples, axis=1)
        mag_ma = moving_average(magnetization, window_size)
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        # Energy convergence
        axes[0].plot(energies, alpha=0.3, linewidth=0.5, label='Raw')
        axes[0].plot(np.arange(len(energy_ma)) + window_size//2, energy_ma,
                    linewidth=2, color='red', label=f'MA({window_size})')
        axes[0].set_ylabel('Energy', fontsize=12)
        axes[0].set_title(f'{title} - Energy Convergence', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Magnetization convergence
        axes[1].plot(magnetization, alpha=0.3, linewidth=0.5, label='Raw')
        axes[1].plot(np.arange(len(mag_ma)) + window_size//2, mag_ma,
                    linewidth=2, color='blue', label=f'MA({window_size})')
        axes[1].set_ylabel('Magnetization', fontsize=12)
        axes[1].set_title(f'{title} - Magnetization Convergence', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
        
        # Variance (as proxy for mixing)
        energy_var = np.array([np.var(energies[max(0, i-window_size):i+1])
                              for i in range(len(energies))])
        axes[2].plot(energy_var, linewidth=1, color='green')
        axes[2].set_xlabel('Sample Index', fontsize=12)
        axes[2].set_ylabel('Energy Variance', fontsize=12)
        axes[2].set_title(f'{title} - Variance (Mixing Quality)', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_blocking_visualization(
        self,
        blocks: List[List[int]],
        grid_shape: Optional[Tuple[int, int]] = None,
        node_positions: Optional[np.ndarray] = None,
        title: str = "Blocking Strategy Visualization"
    ) -> Figure:
        """
        Visualize blocking strategy with color-coded blocks.
        
        Args:
            blocks: List of blocks, each block is a list of node IDs
            grid_shape: Optional (rows, cols) for grid layout
            node_positions: Optional (n_nodes, 2) array of positions
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        # Get total number of nodes
        all_nodes = []
        for block in blocks:
            all_nodes.extend(block)
        n_nodes = len(all_nodes)
        
        # Assign block IDs to nodes
        block_assignment = np.zeros(n_nodes)
        for block_id, block in enumerate(blocks):
            for node_id in block:
                if node_id < n_nodes:
                    block_assignment[node_id] = block_id
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if node_positions is not None and len(node_positions) >= n_nodes:
            # Scatter plot with positions
            scatter = ax.scatter(
                node_positions[:n_nodes, 0],
                node_positions[:n_nodes, 1],
                c=block_assignment,
                cmap='tab20',
                s=100,
                edgecolors='black',
                linewidths=1
            )
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)
            ax.set_aspect('equal')
        else:
            # Grid layout
            if grid_shape is None:
                side = int(np.ceil(np.sqrt(n_nodes)))
                grid_shape = (side, side)
            
            rows, cols = grid_shape
            n_needed = rows * cols
            
            # Pad block_assignment if needed
            if len(block_assignment) < n_needed:
                padded = np.zeros(n_needed)
                padded[:len(block_assignment)] = block_assignment
                block_assignment = padded
            
            grid = block_assignment[:n_needed].reshape(rows, cols)
            
            scatter = ax.imshow(grid, cmap='tab20', interpolation='nearest')
            ax.set_xlabel('Column', fontsize=12)
            ax.set_ylabel('Row', fontsize=12)
            
            # Grid lines
            ax.set_xticks(np.arange(cols) - 0.5, minor=True)
            ax.set_yticks(np.arange(rows) - 0.5, minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Block ID', fontsize=12)
        
        # Add block statistics
        stats_text = f'Blocks: {len(blocks)}\nNodes: {n_nodes}'
        if len(blocks) > 0:
            sizes = [len(b) for b in blocks]
            stats_text += f'\nMin size: {min(sizes)}\nMax size: {max(sizes)}\nMean size: {np.mean(sizes):.1f}'
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_multi_chain_comparison(
        self,
        all_samples: np.ndarray,
        all_energies: np.ndarray,
        title: str = "Multi-Chain Comparison"
    ) -> Figure:
        """
        Compare multiple sampling chains.
        
        Args:
            all_samples: (n_chains, n_samples, n_nodes) array
            all_energies: (n_chains, n_samples) array
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        n_chains, n_samples, n_nodes = all_samples.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Energy trajectories
        for chain_id in range(n_chains):
            axes[0, 0].plot(all_energies[chain_id], alpha=0.7, 
                           linewidth=1, label=f'Chain {chain_id}')
        axes[0, 0].set_xlabel('Sample', fontsize=12)
        axes[0, 0].set_ylabel('Energy', fontsize=12)
        axes[0, 0].set_title(f'{title} - Energy Trajectories', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend(ncol=2, fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy distributions
        for chain_id in range(n_chains):
            axes[0, 1].hist(all_energies[chain_id], bins=30, alpha=0.5,
                           label=f'Chain {chain_id}')
        axes[0, 1].set_xlabel('Energy', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title(f'{title} - Energy Distributions',
                            fontsize=14, fontweight='bold')
        axes[0, 1].legend(ncol=2, fontsize=8)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Magnetization trajectories
        for chain_id in range(n_chains):
            mag = np.mean(all_samples[chain_id], axis=1)
            axes[1, 0].plot(mag, alpha=0.7, linewidth=1, label=f'Chain {chain_id}')
        axes[1, 0].set_xlabel('Sample', fontsize=12)
        axes[1, 0].set_ylabel('Magnetization', fontsize=12)
        axes[1, 0].set_title(f'{title} - Magnetization Trajectories',
                            fontsize=14, fontweight='bold')
        axes[1, 0].legend(ncol=2, fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.5)
        
        # Chain statistics summary
        mean_energies = [np.mean(all_energies[i]) for i in range(n_chains)]
        std_energies = [np.std(all_energies[i]) for i in range(n_chains)]
        
        axes[1, 1].errorbar(range(n_chains), mean_energies, yerr=std_energies,
                           fmt='o', markersize=8, capsize=5, capthick=2)
        axes[1, 1].set_xlabel('Chain ID', fontsize=12)
        axes[1, 1].set_ylabel('Mean Energy ± σ', fontsize=12)
        axes[1, 1].set_title(f'{title} - Chain Statistics',
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_xticks(range(n_chains))
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(
        self,
        samples: np.ndarray,
        energies: np.ndarray,
        blocks: Optional[List[List[int]]] = None,
        grid_shape: Optional[Tuple[int, int]] = None
    ) -> Figure:
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            samples: (n_samples, n_nodes) array of states
            energies: (n_samples,) array of energies
            blocks: Optional list of blocks for visualization
            grid_shape: Optional grid shape
            
        Returns:
            matplotlib Figure with all plots
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # P-bit grid (latest sample)
        ax1 = fig.add_subplot(gs[0, 0])
        if grid_shape is None:
            n_nodes = samples.shape[1]
            side = int(np.ceil(np.sqrt(n_nodes)))
            grid_shape = (side, side)
        rows, cols = grid_shape
        n_needed = rows * cols
        padded = np.zeros(n_needed)
        padded[:len(samples[-1])] = samples[-1]
        grid = padded.reshape(rows, cols)
        im1 = ax1.imshow(grid, cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title('Latest P-bit State')
        plt.colorbar(im1, ax=ax1)
        
        # Energy trajectory
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.plot(energies, linewidth=1)
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy Trajectory')
        ax2.grid(True, alpha=0.3)
        
        # Energy histogram
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(energies, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Energy')
        ax3.set_ylabel('Count')
        ax3.set_title('Energy Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Autocorrelation
        ax4 = fig.add_subplot(gs[1, 1])
        max_lag = min(100, len(samples) - 1)
        lags = np.arange(max_lag + 1)
        autocorr = np.zeros(len(lags))
        mean_sample = np.mean(samples)
        var_sample = np.var(samples)
        if var_sample > 1e-10:
            for lag in lags:
                if lag == 0:
                    autocorr[lag] = 1.0
                else:
                    c = np.mean((samples[:-lag] - mean_sample) * (samples[lag:] - mean_sample))
                    autocorr[lag] = c / var_sample
        ax4.plot(lags, autocorr)
        ax4.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('Autocorrelation')
        ax4.set_title('Autocorrelation')
        ax4.grid(True, alpha=0.3)
        
        # Magnetization
        ax5 = fig.add_subplot(gs[1, 2])
        magnetization = np.mean(samples, axis=1)
        ax5.plot(magnetization, linewidth=1)
        ax5.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax5.set_xlabel('Sample')
        ax5.set_ylabel('Magnetization')
        ax5.set_title('Magnetization')
        ax5.grid(True, alpha=0.3)
        
        # Blocking visualization (if available)
        if blocks is not None:
            ax6 = fig.add_subplot(gs[2, :])
            n_nodes = samples.shape[1]
            block_assignment = np.zeros(n_nodes)
            for block_id, block in enumerate(blocks):
                for node_id in block:
                    if node_id < n_nodes:
                        block_assignment[node_id] = block_id
            padded = np.zeros(n_needed)
            padded[:len(block_assignment)] = block_assignment
            grid_blocks = padded.reshape(rows, cols)
            im6 = ax6.imshow(grid_blocks, cmap='tab20', interpolation='nearest')
            ax6.set_title('Blocking Strategy')
            plt.colorbar(im6, ax=ax6, label='Block ID')
        else:
            ax6 = fig.add_subplot(gs[2, :])
            ax6.text(0.5, 0.5, 'No blocking information available',
                    ha='center', va='center', fontsize=14)
            ax6.axis('off')
        
        fig.suptitle('THRML Comprehensive Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


# ============================================================================
# Convenience Functions
# ============================================================================

def visualize_thrml_state(
    samples: np.ndarray,
    energies: Optional[np.ndarray] = None,
    blocks: Optional[List[List[int]]] = None,
    output_path: Optional[str] = None
) -> Figure:
    """
    Quick visualization of THRML state.
    
    Args:
        samples: (n_samples, n_nodes) or (n_nodes,) array
        energies: Optional (n_samples,) energy array
        blocks: Optional blocking structure
        output_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    if samples.ndim == 1:
        # Single sample, just show grid
        viz = THRMLVisualizer()
        fig = viz.plot_pbit_grid(samples)
    else:
        # Multiple samples, show dashboard
        viz = THRMLVisualizer()
        fig = viz.create_comprehensive_dashboard(samples, energies, blocks)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    
    return fig


if __name__ == '__main__':
    # Demo
    print("THRML Visualizer Demo")
    
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, cannot run demo")
    else:
        # Create synthetic data
        n_samples = 200
        n_nodes = 64
        
        samples = np.random.choice([-1, 1], size=(n_samples, n_nodes))
        energies = -np.sum(samples, axis=1) + np.random.randn(n_samples) * 5
        
        # Visualize
        viz = THRMLVisualizer()
        fig = viz.create_comprehensive_dashboard(samples, energies)
        plt.show()
        
        print("Demo complete!")

