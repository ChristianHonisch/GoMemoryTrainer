#!/usr/bin/env python3
"""
Build script to create executable using PyInstaller.
Run: python build.py
"""

import subprocess
import sys
from pathlib import Path

def build_executable():
    """Build the executable using PyInstaller."""
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Error: PyInstaller not found. Install it with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    project_dir = Path(__file__).parent
    trainer_file = project_dir / "trainer.py"
    
    if not trainer_file.exists():
        print(f"Error: {trainer_file} not found")
        sys.exit(1)
    
    print("Building executable...")
    print(f"Source: {trainer_file}")
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # Single executable file
        "--windowed",                   # No console window
        "--name", "GoMemoryTrainer",    # Output name
        "--distpath", str(project_dir / "dist"),
        "--buildpath", str(project_dir / "build"),
        "--specpath", str(project_dir),
        str(trainer_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        exe_path = project_dir / "dist" / "GoMemoryTrainer.exe"
        print(f"\nSuccess! Executable created at:")
        print(f"  {exe_path}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nError building executable: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(build_executable())
