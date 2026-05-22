"""
Credentials Manager - CRUD operations for credentials using global dotenv storage.
Stores credentials in python/.env and syncs them with os.environ.
"""
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
logger = logging.getLogger(__name__)
# Global .env path  resolved via OPENCHAD_PYTHON_DIR set by python/main.py at boot
_PYTHON_DIR = os.environ.get("OPENCHAD_PYTHON_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python")
)
ENV_PATH = os.path.join(_PYTHON_DIR, ".env")

def _load_env_file() -> Dict[str, str]:
    """Load credentials from the global .env file."""
    credentials = {}
    if not os.path.exists(ENV_PATH):
        return credentials
    try:
        with open(ENV_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    credentials[key] = value
    except Exception as e:
        logger.error(f"Error loading .env file {ENV_PATH}: {e}")
    return credentials

def _save_env_file(credentials: Dict[str, str]) -> bool:
    """Save credentials to the global .env file and sync with os.environ."""
    try:
        with open(ENV_PATH, 'w', encoding='utf-8') as f:
            f.write("# Global Credentials file - Auto-generated\n")
            f.write(f"# Last updated: {datetime.now().isoformat()}\n\n")
            for key, value in sorted(credentials.items()):
                # Sync with os.environ
                os.environ[key] = value
                # Write to file
                # Quote values with spaces or special characters
                if ' ' in value or '=' in value or '"' in value:
                    # Escape internal quotes
                    safe_value = value.replace('"', '\\"')
                    f.write(f"{key}=\"{safe_value}\"\n")
                else:
                    f.write(f"{key}={value}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving .env file {ENV_PATH}: {e}")
        return False

def _mask_value(value: str, show_chars: int = 4) -> str:
    """Mask a credential value for display."""
    if len(value) <= show_chars * 2:
        return '*' * len(value)
    return value[:show_chars] + '*' * (len(value) - show_chars * 2) + value[-show_chars:]

async def initialize_credentials():
    """Initialize credentials by loading them from .env into os.environ."""
    logger.info("Initializing global credentials...")
    credentials = _load_env_file()
    for key, value in credentials.items():
        os.environ[key] = value
    logger.info(f"Loaded {len(credentials)} credentials into environment.")

async def credentials_handler(data: dict) -> dict:
    """
    Handle global credentials operations and return response.
    Commands:
        - list: List all credentials (with masked values)
        - get: Get a specific credential by name
        - add: Add a new credential
        - update: Update an existing credential
        - delete: Delete a credential
        - set: Add or update a credential
    """
    try:
        command = data.get("command")
        credentials = _load_env_file()
        if command == "list":
            # Return all credentials with masked values
            masked_list = []
            for key, value in credentials.items():
                masked_list.append({
                    "name": key,
                    "value": _mask_value(value),
                    "length": len(value)
                })
            return {"data": {"credentials": masked_list, "count": len(masked_list)}}
        elif command == "get":
            name = data.get("name")
            if not name:
                return {"error": "Credential name required"}
            # Try from loaded credentials first, then os.environ
            value = credentials.get(name) or os.environ.get(name)
            if value is None:
                return {"error": f"Credential '{name}' not found"}
            # Option to get unmasked value
            unmask = data.get("unmask", False)
            return {"data": {"name": name, "value": value if unmask else _mask_value(value)}}
        elif command == "add" or command == "update" or command == "set":
            name = data.get("name")
            value = data.get("value")
            if not name:
                return {"error": "Credential name required"}
            if value is None:
                return {"error": "Credential value required"}
            # Normalize name (uppercase, underscores)
            name = name.upper().replace('-', '_').replace(' ', '_')
            is_new = name not in credentials
            if command == "add" and not is_new:
                return {"error": f"Credential '{name}' already exists. Use 'update' to modify."}
            if command == "update" and is_new:
                return {"error": f"Credential '{name}' not found. Use 'add' to create."}
            credentials[name] = str(value)
            if _save_env_file(credentials):
                return {"data": {"status": "added" if is_new else "updated", "name": name}}
            return {"error": "Failed to save credentials"}
        elif command == "delete":
            name = data.get("name")
            if not name:
                return {"error": "Credential name required"}
            # Normalize name
            name = name.upper().replace('-', '_').replace(' ', '_')
            if name not in credentials:
                return {"error": f"Credential '{name}' not found"}
            del credentials[name]
            # Also remove from os.environ
            if name in os.environ:
                del os.environ[name]
            if _save_env_file(credentials):
                return {"data": {"status": "deleted", "name": name}}
            return {"error": "Failed to save credentials"}
        else:
            return {"error": f"Unknown command: {command}"}
    except Exception as e:
        logger.error(f"Error in credentials_handler: {e}", exc_info=True)
        return {"error": str(e)}
