import json
import aiosqlite
import hashlib
import json
import os
import logging
import asyncio
from typing import Dict, Any, Optional
logger = logging.getLogger(__name__)
# Cache for active connections: {db_name: aiosqlite.Connection}
connections: Dict[str, aiosqlite.Connection] = {}
# Cache for database locks: {db_name: asyncio.Lock}
locks: Dict[str, asyncio.Lock] = {}
# Base directory for workspaces
WORKSPACES_DIR = "Workspaces"

def hash_table(input_str: str) -> str:
    """
    Generate consistent 32-character hex ID from string.
    Uses SHA-256 hash truncated to 128 bits (32 hex chars).
    """
    return "tb_" + hashlib.sha256(input_str.encode('utf-8')).hexdigest()[:32]

def get_db_path(db_name: str) -> str:
    """Get the full path for a database file.
    Database files are stored under Workspaces/{db_name}/{db_name}.db
    This ensures each workspace has its own isolated database.
    """
    return os.path.normpath(os.path.join(os.environ.get('OPENCHAD_PROJECT_DIR', '../'), WORKSPACES_DIR, db_name, f"{db_name}.db"))

def get_db_mtime(db_name: str) -> float:
    """Get the modification time of a database file"""
    try:
        db_path = get_db_path(db_name)
        if os.path.exists(db_path):
            return os.path.getmtime(db_path)
    except Exception as e:
        logger.error(f"Error getting db mtime for {db_name}: {e}")
    return 0.0

async def get_connection(db_name: str) -> aiosqlite.Connection:
    """Get or create an async connection to the database"""
    if db_name not in connections:
        db_path = get_db_path(db_name)
        # Ensure the workspace directory exists
        workspace_dir = os.path.dirname(db_path)
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir, exist_ok=True)
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        connections[db_name] = conn
    if db_name not in locks:
        locks[db_name] = asyncio.Lock()
    return connections[db_name]

async def close_all_connections():
    """Close all active database connections (async)"""
    for db_name, conn in list(connections.items()):
        try:
            logger.info(f"Closing database connection for: {db_name}")
            await conn.close()
        except Exception as e:
            logger.error(f"Error closing connection for {db_name}: {e}")
    connections.clear()
    locks.clear()

def quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier (table or column name) to prevent injection and handle special characters."""
    return '"' + identifier.replace('"', '""') + '"'

async def sqlite(data: dict) -> dict:
    """Handle SQLite commands and return response (async)"""
    db_name = data.get("db")
    if not db_name:
        return {"error": "Database name required"}
    # Use a per-database lock to prevent concurrent transaction interference
    if db_name not in locks:
        locks[db_name] = asyncio.Lock()
    async with locks[db_name]:
        conn = await get_connection(db_name)
        try:
            command = data.get("command")
            table_name = data.get("table")
            if command == "query":
                sql = data.get("sql")
                if not sql:
                    logger.error(f"Data: {json.dumps(data)}")
                    return {"error": "SQL query required"}
                params = data.get("params", [])
                try:
                    async with conn.execute(sql, params) as cursor:
                        rows = await cursor.fetchall()
                        result = [dict(row) for row in rows]
                        return {"data": result}
                except aiosqlite.OperationalError as e:
                    if "no such table" in str(e).lower():
                        return {"data": []}
                    logger.error(f"Data: {json.dumps(data)}")
                    raise e
            elif command == "sync_table":
                if not table_name:
                    return {"error": "Table name required"}
                quoted_table = quote_identifier(table_name)
                rows_data: Dict[str, Any] = data.get("data", {})
                # Start transaction for atomicity
                # Python's sqlite3 implicitly starts transactions for DML statements by default,
                # so explicitly calling BEGIN can raise 'cannot start a transaction within a transaction'
                try:
                    # If empty data, clear the table
                    if not rows_data:
                        await conn.execute(f"DROP TABLE IF EXISTS {quoted_table}")
                        await conn.commit()
                        return {"data": {"status": "cleared"}}
                    
                    # Infer columns from the first item
                    first_key = next(iter(rows_data))
                    sample_row = rows_data[first_key]
                    columns = []
                    for k, v in sample_row.items():
                        if k == "id":
                            continue
                        columns.append(f'{quote_identifier(k)}')
                    
                    # Drop and recreate table (fastest way for full sync)
                    await conn.execute(f"DROP TABLE IF EXISTS {quoted_table}")
                    cols_def = ", ".join(columns)
                    if cols_def:
                        cols_def = ", " + cols_def
                    create_sql = f"CREATE TABLE {quoted_table} (id TEXT PRIMARY KEY{cols_def})"
                    await conn.execute(create_sql)

                    # Insert data in batch
                    for row_id, row_content in rows_data.items():
                        keys = ["id"]
                        values = [row_id]
                        placeholders = ["?"]
                        for k, v in row_content.items():
                            if k == "id":
                                continue
                            keys.append(quote_identifier(k))
                            # Always JSON-serialize _v so booleans/numbers round-trip
                            # correctly through SQLite (avoids false -> 0 -> "0" coercion)
                            if k == "_v" or isinstance(v, (dict, list, bool)):
                                v = json.dumps(v)
                            values.append(v)
                            placeholders.append("?")
                        sql = f'INSERT INTO {quoted_table} ({", ".join(keys)}) VALUES ({", ".join(placeholders)})'
                        await conn.execute(sql, values)
                    await conn.commit()
                except Exception as tx_error:
                    await conn.rollback()
                    logger.error(f"Data: {json.dumps(data)}")
                    raise tx_error
                
                # Retrieve final state
                async with conn.execute(f"SELECT * FROM {quoted_table}") as cursor:
                    rows = await cursor.fetchall()
                    result = [dict(row) for row in rows]
                    return {
                        "data": {"status": "synced", "count": len(rows_data), "tables": result}
                    }
            elif command == "execute":
                sql = data.get("sql")
                if not sql:
                    logger.error(f"Data: {json.dumps(data)}")
                    return {"error": "SQL query required"}
                params = data.get("params", [])
                async with conn.execute(sql, params) as cursor:
                    await conn.commit()
                    return {"data": {"changes": cursor.rowcount, "lastrowid": cursor.lastrowid}}
            else:
                logger.error(f"Data: {json.dumps(data)}")
                return {"error": f"Unknown command: {command}"}
        except aiosqlite.Error as e:
            if 'conn' in locals() and conn.in_transaction:
                await conn.rollback()
            logger.error(f"Database error in sqlite handler: {e}", exc_info=True)
            logger.error(f"Data: {json.dumps(data)}")
            return {"error": f"Database error: {str(e)}"}
        except Exception as e:
            if 'conn' in locals() and conn.in_transaction:
                await conn.rollback()
            logger.error(f"Unexpected error in sqlite handler: {e}", exc_info=True)
            logger.error(f"Data: {json.dumps(data)}")
            return {"error": str(e)}
