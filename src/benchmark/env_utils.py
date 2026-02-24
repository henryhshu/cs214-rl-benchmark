# src/benchmark/env_utils.py

import os
import pathlib
import sys

def configure_openenv_envs():
    """
    Configures the environment variables for OpenEnv environments.
    """
    project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
    openenv_path = project_root / "external" / "OpenEnv"
    envs_path = openenv_path / "envs"

    if not envs_path.exists():
        raise FileNotFoundError(f"OpenEnv envs directory not found at {envs_path}")
    
    if openenv_path not in sys.path:
        sys.path.insert(0, str(openenv_path))
