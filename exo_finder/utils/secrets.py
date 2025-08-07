"""
Secret management utility for exo-finder.

This module provides a simple interface for managing secrets using environment variables
and .env files. It allows users to securely configure secrets without storing sensitive
information in plain text within the codebase.
"""

import os
from typing import Optional, Any
from dotenv import load_dotenv
from paths import PROJECT_ROOT


def load_secrets(env_file: Optional[str] = None) -> None:
    """
    Load secrets from a .env file.
    
    Args:
        env_file: Path to the .env file. If None, it will look for a .env file in the project root.
    """
    if env_file is None:
        env_file = PROJECT_ROOT / '.env'
    
    load_dotenv(dotenv_path=env_file)


def get_secret(key: str, default: Any = None) -> Optional[str]:
    """
    Get a secret from environment variables.
    
    Args:
        key: The name of the secret.
        default: Default value if the secret is not found.
        
    Returns:
        The secret value or the default value if not found.
    """
    return os.environ.get(key, default)


# Load secrets automatically when the module is imported
load_secrets()
