#!/usr/bin/env python3
"""
GMCS Environment Setup Script

Helps configure the appropriate environment based on system capabilities.
"""

import subprocess
import sys
import platform
import argparse


def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_rocm_available():
    """Check if ROCm is available (AMD GPUs)."""
    try:
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_cuda_version():
    """Get CUDA version if available."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Extract version like "12.1"
                    parts = line.split('release')
                    if len(parts) > 1:
                        version = parts[1].strip().split(',')[0].strip()
                        return version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def install_requirements(requirements_file):
    """Install requirements from file."""
    print(f"\nInstalling requirements from {requirements_file}...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            check=True
        )
        print(f"✓ Successfully installed {requirements_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing {requirements_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup GMCS environment based on system capabilities"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "gpu", "cpu", "dev", "optional"],
        default="auto",
        help="Installation mode (default: auto-detect)"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional dependencies"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GMCS Environment Setup")
    print("=" * 60)
    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Auto-detect mode
    if args.mode == "auto":
        print("\nDetecting system capabilities...")
        
        has_cuda = check_cuda_available()
        has_rocm = check_rocm_available()
        
        if has_cuda:
            cuda_version = get_cuda_version()
            print(f"✓ CUDA detected: {cuda_version or 'unknown version'}")
            mode = "gpu"
        elif has_rocm:
            print("✓ ROCm detected (AMD GPU)")
            mode = "gpu"
            print("\nNote: ROCm support requires manual configuration.")
            print("Edit requirements-gpu.txt to use ROCm versions.")
        else:
            print("✗ No GPU detected")
            mode = "cpu"
    else:
        mode = args.mode
    
    print(f"\nInstallation mode: {mode.upper()}")
    print("-" * 60)
    
    # Install based on mode
    success = True
    
    if mode == "cpu":
        success &= install_requirements("requirements-cpu.txt")
    
    elif mode == "gpu":
        success &= install_requirements("requirements-gpu.txt")
    
    elif mode == "dev":
        success &= install_requirements("requirements.txt")
        success &= install_requirements("requirements-dev.txt")
    
    elif mode == "optional":
        success &= install_requirements("requirements.txt")
        success &= install_requirements("requirements-optional.txt")
    
    # Install optional dependencies if requested
    if not args.skip_optional and mode in ["gpu", "cpu"]:
        response = input("\nInstall optional dependencies? (y/N): ")
        if response.lower() == 'y':
            install_requirements("requirements-optional.txt")
    
    # Verify installation
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("-" * 60)
    
    # Test JAX
    try:
        import jax
        print(f"✓ JAX {jax.__version__} installed")
        devices = jax.devices()
        print(f"  Available devices: {[str(d) for d in devices]}")
    except ImportError as e:
        print(f"✗ JAX not available: {e}")
        success = False
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.device_count()} device(s)")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  Running on CPU")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} installed")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  GPU devices: {len(gpus)}")
        else:
            print("  Running on CPU")
    except ImportError as e:
        print(f"✗ TensorFlow not available: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ Environment setup complete!")
        print("\nNext steps:")
        print("1. Start backend: python src/main.py")
        print("2. Start frontend: cd frontend && npm install && npm run dev")
        print("3. Open http://localhost:3000")
    else:
        print("✗ Some packages failed to install")
        print("\nTroubleshooting:")
        print("1. Ensure Python 3.10+ is installed")
        print("2. For GPU support, install NVIDIA drivers and CUDA toolkit")
        print("3. Check individual error messages above")
        print("4. See docs/ for detailed installation guides")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

