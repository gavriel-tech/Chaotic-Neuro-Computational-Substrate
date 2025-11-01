#!/usr/bin/env python3
"""
GMCS Test Runner

Runs test suite with proper configuration and reports results.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_pytest():
    """Install pytest if missing."""
    try:
        import pytest
        return True
    except ImportError:
        print("pytest not found, installing...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-cov"],
            capture_output=True
        )
        return result.returncode == 0


def run_tests(args=None):
    """Run pytest with arguments."""
    if args is None:
        args = []
    
    # Set PYTHONPATH to include project root
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    # Also add THRML to path
    thrml_path = Path.cwd() / 'thrml-main' / 'thrml-main'
    if thrml_path.exists():
        os.environ['PYTHONPATH'] += os.pathsep + str(thrml_path)
    
    # Run pytest
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"] + args
    print(f"Running: {' '.join(cmd)}")
    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
    print()
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main test runner."""
    print("=" * 60)
    print("GMCS Test Suite")
    print("=" * 60)
    print()
    
    if not install_pytest():
        print("Failed to install pytest")
        return 1
    
    # Parse arguments
    test_args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/"]
    
    return run_tests(test_args)


if __name__ == '__main__':
    sys.exit(main())

