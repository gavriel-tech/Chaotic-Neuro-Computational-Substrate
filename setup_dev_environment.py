#!/usr/bin/env python3
"""
GMCS Development Environment Setup

This script verifies and sets up the development environment,
installing missing dependencies and running basic checks.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def check_python_version():
    """Verify Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  [FAIL] Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_jax():
    """Verify JAX installation."""
    print("Checking JAX...")
    try:
        import jax
        print(f"  [OK] JAX {jax.__version__}")
        
        # Check for GPU
        devices = jax.devices()
        has_gpu = any("gpu" in str(d).lower() for d in devices)
        if has_gpu:
            n_gpus = len([d for d in devices if 'gpu' in str(d).lower()])
            print(f"  [OK] GPU acceleration available ({n_gpus} GPU(s))")
        else:
            print(f"  [WARN] CPU only (no GPU detected)")
        
        return True
    except ImportError:
        print("  [FAIL] JAX not installed")
        return False


def check_thrml():
    """Verify THRML installation."""
    print("Checking THRML...")
    try:
        sys.path.insert(0, str(Path('thrml-main/thrml-main').absolute()))
        import thrml
        print(f"  [OK] THRML installed")
        return True
    except ImportError:
        print("  [FAIL] THRML not found")
        return False


def check_pytest():
    """Check if pytest is installed."""
    print("Checking pytest...")
    try:
        import pytest
        print(f"  [OK] pytest {pytest.__version__}")
        return True
    except ImportError:
        print("  [FAIL] pytest not installed")
        return False


def check_frontend():
    """Verify frontend dependencies."""
    print("Checking frontend...")
    frontend_dir = Path('frontend')
    if not frontend_dir.exists():
        print("  [FAIL] frontend/ directory not found")
        return False
    
    node_modules = frontend_dir / 'node_modules'
    if not node_modules.exists():
        print("  [WARN] node_modules not found, running npm install...")
        success, _, _ = run_command("cd frontend && npm install")
        if success:
            print("  [OK] Frontend dependencies installed")
            return True
        else:
            print("  [FAIL] npm install failed")
            return False
    else:
        print("  [OK] Frontend dependencies installed")
        return True


def install_python_deps():
    """Install missing Python dependencies."""
    print("\nInstalling Python dependencies...")
    success, _, stderr = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if not success:
        print(f"  [WARN] Core dependencies may have failed: {stderr[:200]}")

    dev_success, _, dev_stderr = run_command(f"{sys.executable} -m pip install -r requirements-dev.txt")
    if not dev_success:
        print(f"  [WARN] Dev dependencies may have failed: {dev_stderr[:200]}")

    if success and dev_success:
        print("  [OK] Core and dev dependencies installed")
    else:
        print("  [WARN] Some dependencies reported installation issues; review logs above.")

    return True  # Continue anyway


def install_pytest():
    """Install pytest and testing tools."""
    print("\nInstalling pytest...")
    success, _, stderr = run_command(
        f"{sys.executable} -m pip install pytest pytest-asyncio pytest-cov"
    )
    if success:
        print("  [OK] pytest installed")
        return True
    else:
        print(f"  [FAIL] pytest installation failed: {stderr}")
        return False


def run_basic_test():
    """Run basic functionality test."""
    print("\nRunning basic functionality test...")
    
    # Create test script
    test_script = Path('_test_basic.py')
    test_script.write_text('''
import sys
sys.path.insert(0, '.')
from src.core.thrml_integration import create_thrml_model
import numpy as np
import jax.random

# Create model
wrapper = create_thrml_model(
    n_nodes=10,
    weights=np.eye(10) * 0.1,
    biases=np.zeros(10),
    beta=1.0
)

# Sample
key = jax.random.PRNGKey(0)
samples = wrapper.sample_gibbs(n_steps=5, temperature=1.0, key=key)

print(f"[OK] Basic test passed: sampled {len(samples)} nodes")
''')
    
    success, stdout, stderr = run_command(f"{sys.executable} _test_basic.py", check=False)
    test_script.unlink()  # Clean up
    
    if success:
        print(f"  {stdout.strip()}")
        return True
    else:
        print(f"  [FAIL] Test failed: {stderr[:500]}")
        return False


def main():
    """Main setup routine."""
    print("=" * 60)
    print("GMCS Development Environment Setup")
    print("=" * 60)
    print()
    
    checks = []
    
    # Run checks
    checks.append(("Python", check_python_version()))
    checks.append(("JAX", check_jax()))
    checks.append(("THRML", check_thrml()))
    checks.append(("pytest", check_pytest()))
    checks.append(("Frontend", check_frontend()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\n[WARN] Some checks failed. Attempting to fix...")
        
        # Install missing dependencies
        if not checks[3][1]:  # pytest missing
            install_pytest()
        
        if not checks[1][1] or not checks[2][1]:  # JAX or THRML missing
            install_python_deps()
        
        if not checks[4][1]:  # Frontend missing
            check_frontend()
    
    # Final test
    print()
    if run_basic_test():
        print("\n" + "=" * 60)
        print("[SUCCESS] Environment setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Start backend: python src/main.py")
        print("  2. Start frontend: cd frontend && npm run dev")
        print("  3. Open browser: http://localhost:3000")
        print("\nFor help, see COMPREHENSIVE_REVIEW_AND_IMPROVEMENTS.md")
        return 0
    else:
        print("\n" + "=" * 60)
        print("[FAIL] Setup incomplete")
        print("=" * 60)
        print("\nSome tests failed. Common issues:")
        print("  - JAX/CUDA version mismatch")
        print("  - THRML not in correct location")
        print("  - Missing system dependencies")
        print("\nSee COMPREHENSIVE_REVIEW_AND_IMPROVEMENTS.md for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())

