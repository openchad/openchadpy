"""
Settings - Centralized settings management with SQLite persistence.
Settings are defined per plugin source via settings.toml files
"""
import aiosqlite
import json
import os
import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable
logger = logging.getLogger(__name__)
# Supported setting types
SETTING_TYPES = {"string", "int", "float", "boolean", "array", "object"}
# Plugin source directories that can have settings.toml
PLUGIN_SOURCES = [
    "Apps",
    "Backend",
    "Pipeline",
    "Tools",
    "ModelProvider",
]

def _infer_type(value: Any) -> str:
    """Infer the setting type from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"

def _cast_value(value: Any, setting_type: str) -> Any:
    """Cast a value to the correct Python type based on setting_type."""
    if setting_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if setting_type == "int":
        return int(value)
    if setting_type == "float":
        return float(value)
    if setting_type == "array":
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [v.strip() for v in value.split(",") if v.strip()]
        return [value] if value else []
    if setting_type == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return {} if value is not None else None
    # string
    return str(value) if value is not None else ""

def _serialize_value(value: Any) -> str:
    """Serialize a value to JSON string for storage."""
    return json.dumps(value)

def _deserialize_value(raw: str, setting_type: str) -> Any:
    """Deserialize a stored JSON string back to a typed value."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        parsed = raw
    return _cast_value(parsed, setting_type)
class Settings:
    """
    Centralized settings manager with global SQLite persistence.
    """
    
    def __init__(self, project_root: str, on_change: Optional[Callable[[str], Awaitable[None]]] = None):
        self.project_root = project_root
        self._callbacks: List[Callable[[str], Awaitable[None]]] = []
        if on_change:
            self._callbacks.append(on_change)
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        # In-memory cache of TOML defaults
        self._defaults: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self):
        """Pre-load TOML defaults from the filesystem and sync to DB."""
        async with self._lock:
            await self._load_toml_defaults()
            self._initialized = False
        await self._ensure_ready()
        logger.info(f"Settings base initialized with {len(self._defaults)} defaults from TOML files")

    def _get_db_path(self) -> str:
        """Get the absolute path to the settings database."""
        return os.path.normpath(os.path.join(os.environ.get("OPENCHAD_PROJECT_DIR", self.project_root), "settings.db"))

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create a database connection."""
        async with self._lock:
            if self._connection is None:
                db_path = self._get_db_path()
                db_dir = os.path.dirname(db_path)
                if not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                self._connection = await aiosqlite.connect(db_path)
                self._connection.row_factory = aiosqlite.Row
            return self._connection

    async def _ensure_ready(self):
        """Ensure the table exists and defaults are loaded."""
        if self._initialized:
            return
        conn = await self._get_connection()
        async with self._lock:
            # Re-check after acquiring lock
            if self._initialized:
                return
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'string',
                    source TEXT NOT NULL DEFAULT '',
                    section TEXT NOT NULL DEFAULT '',
                    default_value TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            await conn.commit()
            # Load defaults if not present
            now = datetime.now(timezone.utc).isoformat()
            changed_keys = []
            for full_key, info in self._defaults.items():
                async with conn.execute(
                    "SELECT key, default_value FROM settings WHERE key = ?", (full_key,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row is None:
                        await conn.execute(
                            """INSERT INTO settings (key, value, type, source, section, default_value, updated_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (full_key, info["serialized"], info["type"], info["source"], info["section"], info["serialized"], now),
                        )
                    elif row["default_value"] != info["serialized"]:
                        await conn.execute(
                            "UPDATE settings SET value = ?, default_value = ?, updated_at = ? WHERE key = ?",
                            (info["serialized"], info["serialized"], now, full_key),
                        )
                        changed_keys.append(full_key)
            await conn.commit()
            self._initialized = True
        # Trigger callbacks for updated keys outside the lock
        for key in changed_keys:
            for callback in self._callbacks:
                try:
                    await callback(key)
                except Exception as e:
                    logger.error(f"Error in settings callback for {key}: {e}")

    def _parse_toml_simple(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        """Parse TOML file using built-in tomllib."""
        import tomllib
        try:
            with open(filepath, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.error(f"Failed to parse TOML file {filepath}: {e}")
            return {}

    async def _load_toml_defaults(self):
        """Scan filesystem for default settings."""
        self._defaults.clear()
        for source_name in PLUGIN_SOURCES:
            source_dir = os.path.join(self.project_root, source_name)
            if not os.path.isdir(source_dir):
                continue
            # Check root-level settings.toml
            root_toml = os.path.join(source_dir, "settings.toml")
            if os.path.isfile(root_toml):
                self._cache_toml_file(root_toml, source_name)
            # Check plugin-level settings.toml (e.g. ModelProvider/publisher/plugin/settings.toml)
            for publisher in os.listdir(source_dir):
                publisher_dir = os.path.join(source_dir, publisher)
                if not os.path.isdir(publisher_dir) or publisher.startswith(('.', '_')):
                    continue
                for plugin_name in os.listdir(publisher_dir):
                    plugin_dir = os.path.join(publisher_dir, plugin_name)
                    if not os.path.isdir(plugin_dir) or plugin_name.startswith(('.', '_')):
                        continue
                    toml_path = os.path.join(plugin_dir, "settings.toml")
                    if os.path.isfile(toml_path):
                        key_source = f"{publisher}/{plugin_name}"
                        self._cache_toml_file(toml_path, key_source)
        settings_dir = os.path.join(self.project_root, "Settings")
        if os.path.isdir(settings_dir):
            for filename in os.listdir(settings_dir):
                if filename.endswith(".toml"):
                    file_stem = os.path.splitext(filename)[0]
                    self._cache_toml_file(os.path.join(settings_dir, filename), f"Others/{file_stem}")

    def _cache_toml_file(self, filepath: str, source: str):
        """Parse TOML and store in memory cache."""
        sections = self._parse_toml_simple(filepath)
        for section_name, entries in sections.items():
            for key_name, value in entries.items():
                full_key = f"{source}/{section_name}.{key_name}"
                setting_type = _infer_type(value)
                serialized = _serialize_value(value)
                self._defaults[full_key] = {
                    "type": setting_type,
                    "source": source,
                    "section": section_name,
                    "serialized": serialized,
                }
    # =========================================================================
    # Public API
    # =========================================================================

    async def get(self, key: str) -> Optional[Any]:
        await self._ensure_ready()
        conn = await self._get_connection()
        async with self._lock:
            async with conn.execute(
                "SELECT value, type FROM settings WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
                return _deserialize_value(row["value"], row["type"]) if row else None

    async def get_all(self) -> List[Dict[str, Any]]:
        await self._ensure_ready()
        conn = await self._get_connection()
        async with self._lock:
            async with conn.execute(
                "SELECT key, value, type, source, section, default_value, updated_at FROM settings ORDER BY source, section, key"
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "key": r["key"],
                        "value": _deserialize_value(r["value"], r["type"]),
                        "type": r["type"],
                        "source": r["source"],
                        "section": r["section"],
                        "default_value": _deserialize_value(r["default_value"], r["type"]) if r["default_value"] else None,
                        "updated_at": r["updated_at"],
                    }
                    for r in rows
                ]

    async def set(self, key: str, value: Any) -> bool:
        await self._ensure_ready()
        conn = await self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            async with conn.execute(
                "SELECT type FROM settings WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                logger.warning(f"Undefined setting key: {key}")
                return False
            setting_type = row["type"]
            serialized = _serialize_value(_cast_value(value, setting_type))
            await conn.execute(
                "UPDATE settings SET value = ?, updated_at = ? WHERE key = ?",
                (serialized, now, key),
            )
            await conn.commit()
            if self._callbacks:
                for callback in self._callbacks:
                    try:
                        await callback(key)
                    except Exception as e:
                        logger.error(f"Error in settings callback for {key}: {e}")
            return True

    async def reset(self, key: str) -> bool:
        await self._ensure_ready()
        conn = await self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            async with conn.execute(
                "SELECT default_value FROM settings WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None or row["default_value"] is None:
                    return False
            await conn.execute(
                "UPDATE settings SET value = default_value, updated_at = ? WHERE key = ?",
                (now, key),
            )
            await conn.commit()
            if self._callbacks:
                for callback in self._callbacks:
                    try:
                        await callback(key)
                    except Exception as e:
                        logger.error(f"Error in settings callback for {key}: {e}")
            return True

    async def get_sources(self) -> List[str]:
        await self._ensure_ready()
        conn = await self._get_connection()
        async with self._lock:
            async with conn.execute(
                "SELECT DISTINCT source FROM settings ORDER BY source"
            ) as cursor:
                rows = await cursor.fetchall()
                return [r["source"] for r in rows]

    def subscribe(self, callback: Callable[[str], Awaitable[None]]):
        """Add a callback to be notified when a setting changes."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[str], Awaitable[None]]):
        """Remove a change notification callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def close(self):
        async with self._lock:
            if self._connection:
                try:
                    await self._connection.close()
                except Exception as e:
                    logger.error(f"Error closing settings db: {e}")
                self._connection = None
