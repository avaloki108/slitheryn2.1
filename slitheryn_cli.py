#!/usr/bin/env python3
"""
Slitheryn wrapper script that automatically activates uv venv and loads environment variables.
"""

import os
import sys
import subprocess
from pathlib import Path

def load_env_file(env_file_path):
    """Load environment variables from .env.local file."""
    if env_file_path.exists():
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
        print(f"Loaded environment variables from {env_file_path}")
    else:
        print(f"Environment file {env_file_path} not found")

def main():
    """Main wrapper function."""
    # Get the project directory
    project_dir = Path(__file__).parent
    env_file = project_dir / ".env.local"
    venv_dir = project_dir / ".venv"
    
    # Load environment variables from .env.local
    load_env_file(env_file)
    
    # Check if uv venv exists
    if not venv_dir.exists():
        print("uv venv not found. Creating one...")
        subprocess.run(["uv", "venv"], cwd=project_dir, check=True)
    
    # Get the python executable from uv venv
    python_exe = venv_dir / "bin" / "python"
    if not python_exe.exists():
        print("uv venv python not found. Creating venv...")
        subprocess.run(["uv", "venv"], cwd=project_dir, check=True)
    
    # Ensure slitheryn is installed in the venv
    try:
        subprocess.run([str(python_exe), "-c", "import slitheryn"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Installing slitheryn in uv venv...")
        subprocess.run(["uv", "pip", "install", "-e", "."], cwd=project_dir, check=True)
    
    # Run slitheryn with all arguments passed through
    slitheryn_main = project_dir / "slitheryn" / "__main__.py"
    cmd = [str(python_exe), str(slitheryn_main)] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, cwd=project_dir)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()