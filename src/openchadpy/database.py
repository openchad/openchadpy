"""
Database - High-level CRUD interface for tools.
Provides easy database access that syncs with frontend useDatabase hook.
- Auto-notifies clients on changes
- Uses same hashing as frontend for table names
"""
import hashlib
import json
import logging
from typing import Any, Dict, Optional
from .sqlite import sqlite, get_connection
from .database_manager import trigger_table_update
logger = logging.getLogger(__name__)

def generate_id_from_string(input_str: str) -> str:
    """Generate consistent 32-character hex ID from string (matches frontend)."""
    return "tb_" + hashlib.sha256(input_str.encode('utf-8')).hexdigest()[:32]

class Database:
    """
    High-level database interface for tools.
    Usage in a tool:
        async def execute(self, **kwargs):
            # Set value
            await self.database.set("tasks", "task_1", {"title": "My Task"})
            # Get value
            task = await self.database.get("tasks", "task_1")
            # Get all
            all_tasks = await self.database.get("tasks")
            # Delete
            await self.database.delete("tasks", "task_1")
    """

    def __init__(self, workspace: str = "Private", tab_id: str = ""):
        """
        Initialize database for a workspace.
        Args:
            workspace: Database/workspace name (e.g., "Private")
            tab_id: App name for table namespacing (e.g., "my_app")
        """
        self.workspace = workspace
        self.tab_id = tab_id

    def _get_hashed_table(self, table: str) -> str:
        """Get hashed table name using app/table pattern."""
        prefix = self.tab_id if self.tab_id else ""
        full_name = f"{prefix}/{table}" if prefix else table
        return generate_id_from_string(full_name)
  
    async def get(self, table: str, key: Optional[str] = None) -> Any:
        """
        Get value(s) from table.
        Args:
            table: Table name (will be hashed with tab_id prefix)
            key: Optional key. If None, returns all rows as dict.
        Returns:
            Single value if key provided, else dict of all values
        """
        hashed = self._get_hashed_table(table)
        if key:
            result = await sqlite({
                "db": self.workspace,
                "table": hashed,
                "command": "query",
                "sql": f"SELECT * FROM {hashed} WHERE id = ?",
                "params": [key]
            })
        else:
            result = await sqlite({
                "db": self.workspace,
                "table": hashed,
                "command": "query",
                "sql": f"SELECT * FROM {hashed}"
            })
        if result.get("error"):
            return None if key else {}
        rows = result.get("data", [])
        if key:
            if not rows:
                return None
            row = rows[0]
            # Unwrap _v if present
            if "_v" in row:
                val = row["_v"]
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except:
                        return val
                return val
            return {k: v for k, v in row.items() if k != "id"}
        else:
            # Return as dict keyed by id
            data = {}
            for row in rows:
                row_id = row.get("id")
                if row_id:
                    if "_v" in row:
                        val = row["_v"]
                        if isinstance(val, str):
                            try:
                                data[row_id] = json.loads(val)
                            except:
                                data[row_id] = val
                        else:
                            data[row_id] = val
                    else:
                        data[row_id] = {k: v for k, v in row.items() if k != "id"}
            return data

    async def set(self, table: str, key: str, value: Any) -> bool:
        """
        Set a value in the table (upsert).
        Args:
            table: Table name
            key: Row key/id
            value: Value to store (will be wrapped in _v for non-objects)
        Returns:
            True on success
        """
        hashed = self._get_hashed_table(table)
        # Get existing data
        existing = await self.get(table)
        if existing is None:
            existing = {}
        # Add/update the key
        existing[key] = value
        # Prepare sync payload (wrap values)
        # Match frontend's processDataForSync: pass raw values in _v,
        # the SQLite layer handles JSON serialization for all _v values.
        values = list(existing.values())
        has_primitive = any(
            not isinstance(v, dict) or isinstance(v, list) or v is None
            for v in values
        )
        payload = {}
        for k, v in existing.items():
            if has_primitive or not isinstance(v, dict) or isinstance(v, list):
                payload[k] = {"_v": v}
            else:
                payload[k] = v
        result = await sqlite({
            "db": self.workspace,
            "table": hashed,
            "command": "sync_table",
            "data": payload
        })
        if not result.get("error"):
            await trigger_table_update(self.workspace, hashed)
            return True
        logger.error(f"Database.set error: {result.get('error')}")
        return False

    async def delete(self, table: str, key: str) -> bool:
        """
        Delete a key from the table.
        Args:
            table: Table name
            key: Row key/id to delete
        Returns:
            True on success
        """
        hashed = self._get_hashed_table(table)
        # Get existing data
        existing = await self.get(table)
        if not existing or key not in existing:
            return True  # Already doesn't exist
        # Remove the key
        del existing[key]
        # Re-sync the table
        if not existing:
            # Empty - clear table
            result = await sqlite({
                "db": self.workspace,
                "table": hashed,
                "command": "sync_table",
                "data": {}
            })
        else:
            # Prepare sync payload
            values = list(existing.values())
            has_primitive = any(
                not isinstance(v, dict) or isinstance(v, list) or v is None
                for v in values
            )
            payload = {}
            for k, v in existing.items():
                if has_primitive or not isinstance(v, dict) or isinstance(v, list):
                    payload[k] = {"_v": v}
                else:
                    payload[k] = v
            result = await sqlite({
                "db": self.workspace,
                "table": hashed,
                "command": "sync_table",
                "data": payload
            })
        if not result.get("error"):
            await trigger_table_update(self.workspace, hashed)
            return True
        logger.error(f"Database.delete error: {result.get('error')}")
        return False

    async def sync(self, table: str, data: Any) -> bool:
        """
        Sync entire table data (replaces all existing data).
        Args:
            table: Table name
            data: Dict of {key: value} to sync
        Returns:
            True on success
        """
        hashed = self._get_hashed_table(table)
        # Prepare sync payload
        # Match frontend's processDataForSync: pass raw values in _v,
        # the SQLite layer handles JSON serialization for all _v values.
        values = list(data.values())
        has_primitive = any(
            not isinstance(v, dict) or isinstance(v, list) or v is None
            for v in values
        )
        payload = {}
        for k, v in data.items():
            if has_primitive or not isinstance(v, dict) or isinstance(v, list):
                payload[k] = {"_v": v}
            else:
                payload[k] = v
        result = await sqlite({
            "db": self.workspace,
            "table": hashed,
            "command": "sync_table",
            "data": payload
        })
        if not result.get("error"):
            await trigger_table_update(self.workspace, hashed)
            return True
        logger.error(f"Database.sync error: {result.get('error')}")
        return False
    
    async def query(self, table: str, sql: str) -> Any:
        """
        Execute raw SQL query on a table.
        Args:
            table: Table name (for hashing)
            sql: SQL query (use {table} placeholder for table name)
        Returns:
            Query result
        """
        hashed = self._get_hashed_table(table)
        # Replace {table} placeholder
        sql = sql.replace("{table}", hashed)
        result = await sqlite({
            "db": self.workspace,
            "table": hashed,
            "command": "query",
            "sql": sql
        })
        return result.get("data", [])
