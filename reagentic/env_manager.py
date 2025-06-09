import os
import sys


def _parse_env_file(file_path: str, var_name: str) -> str | None:
    """Parses a .env file and returns the value of a specific variable."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if key == var_name:
                    return value.strip()
    return None


def get_env_var(var_name: str) -> str:
    """
    Retrieves an environment variable, checking .env files if not found in the environment.

    Args:
        var_name: The name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set.
    """
    # 1) check env
    value = os.getenv(var_name)
    if value is not None:
        return value

    # 2) if not found in env, check current folder for .env file
    current_dir_env = _parse_env_file('.env', var_name)
    if current_dir_env is not None:
        return current_dir_env

    # 3) if not found in current folder .env and if interpreter is running from venv,
    # then search for .env in folder where venv folder is placed
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = sys.prefix
        parent_dir_env = _parse_env_file(os.path.join(os.path.dirname(venv_path), '.env'), var_name)
        if parent_dir_env is not None:
            return parent_dir_env

    raise ValueError(f"Environment variable '{var_name}' is not set.")
