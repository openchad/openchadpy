import asyncio
import logging
import time
from typing import Dict, Set
from fastapi import WebSocket
from .connection_manager import manager
logger = logging.getLogger(__name__)
# subscription tracking: {conn_id: set of "db.table" subscriptions}
db_subscriptions: Dict[str, Set[str]] = {}
# Reverse mapping: {"db.table": set of conn_ids}
db_watchers: Dict[str, Set[str]] = {}
# Per-table timestamps - only updated when that specific table changes
db_table_timestamps: Dict[str, float] = {}

async def subscribe_db(conn_id: str, db_name: str, table_name: str):
    """Subscribe a websocket to database table changes"""
    db_table = f"{db_name}.{table_name}"
    # Add to subscriptions
    if conn_id not in db_subscriptions:
        db_subscriptions[conn_id] = set()
    db_subscriptions[conn_id].add(db_table)
    # Add to watchers
    if db_table not in db_watchers:
        db_watchers[db_table] = set()
        # Initialize timestamp if not exists
        if db_table not in db_table_timestamps:
            db_table_timestamps[db_table] = time.time()
    db_watchers[db_table].add(conn_id)
    logger.info(
        f"✓ Subscribed {conn_id} to {db_table} (total watchers: {len(db_watchers[db_table])})"
    )
    return {"status": "subscribed", "db": db_name, "table": table_name}

async def unsubscribe_db(conn_id: str, db_name: str, table_name: str):
    """Unsubscribe a websocket from database table changes"""
    db_table = f"{db_name}.{table_name}"
    if conn_id in db_subscriptions:
        db_subscriptions[conn_id].discard(db_table)
        if not db_subscriptions[conn_id]:
            del db_subscriptions[conn_id]
    if db_table in db_watchers:
        db_watchers[db_table].discard(conn_id)
        remaining_watchers = len(db_watchers[db_table])
        if not db_watchers[db_table]:
            del db_watchers[db_table]
            if db_table in db_table_timestamps:
                del db_table_timestamps[db_table]
            logger.info(f"✗ Unsubscribed {conn_id} from {db_table} (no watchers left)")
        else:
            logger.info(
                f"✗ Unsubscribed {conn_id} from {db_table} ({remaining_watchers} watcher(s) remaining)"
            )
    return {"status": "unsubscribed", "db": db_name, "table": table_name}

async def notify_db_change(db_table: str, timestamp: float):
    """Notify all subscribers of a database change"""
    if db_table not in db_watchers:
        return
    from .event_emitter import event_emitter
    event_name = f"db_changed:{db_table}"
    data = {"timestamp": int(timestamp * 1000)}  # Convert to milliseconds
    conn_ids = list(db_watchers[db_table])
    logger.info(f"🔔 DB changed: {db_table} - notifying {len(conn_ids)} subscriber(s)")
    # Use event_emitter to broadcast to both Tauri and WebSocket clients
    # For WebSocket, we only target specific subscribers
    disconnected = await event_emitter.emit(event_name, data, conn_ids=conn_ids)
    # Clean up disconnected websockets from subscription lists
    cleanup_disconnected_db(disconnected)

def cleanup_disconnected_db(disconnected_ids: list[str]):
    """Remove disconnected websockets from database subscriptions"""
    for conn_id in disconnected_ids:
        if conn_id in db_subscriptions:
            for dt in db_subscriptions[conn_id]:
                if dt in db_watchers:
                    db_watchers[dt].discard(conn_id)
                    # If no more watchers for this table, clean up
                    if not db_watchers[dt]:
                        del db_watchers[dt]
                        if dt in db_table_timestamps:
                            del db_table_timestamps[dt]
            del db_subscriptions[conn_id]

def remove_id(conn_id: str):
    """Clean up subscriptions for a disconnected websocket ID"""
    cleanup_disconnected_db([conn_id])

async def trigger_table_update(db_name: str, table_name: str):
    """Explicitly trigger an update notification for a table"""
    db_table = f"{db_name}.{table_name}"
    timestamp = time.time()
    db_table_timestamps[db_table] = timestamp
    await notify_db_change(db_table, timestamp)
