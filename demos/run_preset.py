"""
Generic Preset Runner for GMCS.

Loads any preset JSON file and executes it using the node graph system.
This demonstrates that the full chain from preset -> nodes -> execution works.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nodes.node_executor import PresetExecutor


def load_preset(preset_path: str) -> dict:
    """Load a preset JSON file."""
    with open(preset_path, 'r') as f:
        preset = json.load(f)
    return preset


def run_preset(preset_path: str, num_steps: int = 100, verbose: bool = True):
    """
    Load and run a preset for N steps.
    
    Args:
        preset_path: Path to preset JSON file
        num_steps: Number of simulation steps
        verbose: Print detailed output
    """
    # Load preset
    if verbose:
        print(f"Loading preset from: {preset_path}")
    
    preset = load_preset(preset_path)
    
    if verbose:
        print(f"Preset: {preset['name']}")
        print(f"Description: {preset.get('description', 'N/A')}")
        print(f"Category: {preset.get('category', 'N/A')}")
        print(f"Nodes: {len(preset['nodes'])}")
        print(f"Connections: {len(preset['connections'])}")
        print()
    
    # Create executor
    executor = PresetExecutor()
    
    try:
        # Load preset into executor
        executor.load_preset(preset)
        
        if verbose:
            print(f"Preset loaded successfully!")
            print(f"Execution order: {' -> '.join(executor.graph.execution_order[:5])}")
            if len(executor.graph.execution_order) > 5:
                print(f"  ... and {len(executor.graph.execution_order) - 5} more nodes")
            print()
        
        # Run simulation
        if verbose:
            print(f"Running for {num_steps} steps...")
            print("-" * 60)
        
        for step in range(num_steps):
            outputs = executor.run_step()
            
            if verbose and step % 10 == 0:
                # Print status every 10 steps
                num_outputs = sum(len(node_out) for node_out in outputs.values())
                print(f"Step {step:4d}/{num_steps}: {num_outputs} outputs from {len(outputs)} nodes")
            
            # Check for errors (all outputs empty might indicate problems)
            if step > 10 and all(len(out) == 0 for out in outputs.values()):
                if verbose:
                    print(f"[WARNING] No outputs after step {step}, stopping early")
                break
        
        if verbose:
            print("-" * 60)
            print(f"✓ Completed {min(step + 1, num_steps)} steps successfully")
            print()
        
        # Get final visualization data
        viz_data = executor.get_visualization_data()
        if verbose:
            print("Final Statistics:")
            print(f"  Total nodes: {viz_data['num_nodes']}")
            print(f"  Total connections: {viz_data['num_connections']}")
            print(f"  Nodes with outputs: {len([k for k, v in viz_data['node_outputs'].items() if v])}")
            print()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Preset execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run a GMCS preset')
    parser.add_argument('preset', type=str, nargs='?', 
                       help='Path to preset JSON file or preset name (e.g., "emergent_logic")')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of steps to run (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--list', action='store_true',
                       help='List available presets')
    
    args = parser.parse_args()
    
    # Find presets directory
    project_root = Path(__file__).parent.parent
    presets_dir = project_root / 'frontend' / 'presets'
    
    if args.list:
        # List available presets
        print("Available presets:")
        print("-" * 60)
        for preset_file in sorted(presets_dir.glob('*.json')):
            try:
                preset = load_preset(str(preset_file))
                print(f"  {preset['id']:20s} - {preset.get('name', 'N/A')}")
            except:
                pass
        return
    
    if not args.preset:
        print("Error: Please specify a preset to run")
        print("Usage: python run_preset.py <preset_name_or_path>")
        print("   or: python run_preset.py --list")
        sys.exit(1)
    
    # Determine preset path
    if os.path.exists(args.preset):
        preset_path = args.preset
    else:
        # Try to find preset by name
        preset_name = args.preset.replace('.json', '')
        preset_path = presets_dir / f'{preset_name}.json'
        
        if not preset_path.exists():
            print(f"Error: Preset not found: {args.preset}")
            print(f"Looked in: {preset_path}")
            print("\nUse --list to see available presets")
            sys.exit(1)
    
    # Run the preset
    print("=" * 60)
    print("GMCS Preset Runner")
    print("=" * 60)
    print()
    
    success = run_preset(
        str(preset_path),
        num_steps=args.steps,
        verbose=not args.quiet
    )
    
    if success:
        print("✓ Preset executed successfully!")
        sys.exit(0)
    else:
        print("✗ Preset execution failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

