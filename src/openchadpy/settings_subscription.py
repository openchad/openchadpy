import asyncio
import logging
import time
from typing import Set
logger = logging.getLogger(__name__)
# Track connection IDs that are subscribed to settings changes
settings_watchers: Set[str] = set()

async def subscribe_settings(conn_id: str):
    """Subscribe a websocket to settings changes"""
    settings_watchers.add(conn_id)
    logger.info(f"✓ Subscribed {conn_id} to settings (watchers: {len(settings_watchers)})")
    return {"status": "subscribed"}

async def unsubscribe_settings(conn_id: str):
    """Unsubscribe a websocket from settings changes"""
    settings_watchers.discard(conn_id)
    logger.info(f"✗ Unsubscribed {conn_id} from settings")
    return {"status": "unsubscribed"}

async def notify_settings_change(key: str):
    """Notify all subscribers that a setting has changed"""
    from .event_emitter import event_emitter, is_tauri_mode
    event_name = "settings_changed"
    data = {
        "key": key,
        "timestamp": int(time.time() * 1000)
    }
    logger.info(f"🔔 Settings changed: {key} - notifying {len(settings_watchers)} subscriber(s)")
    if is_tauri_mode():
        # In Tauri mode there is no per-connection routing  emit to the webview directly
        await event_emitter.emit(event_name, data)
    else:
        # WebSocket mode: only notify subscribed connections
        if not settings_watchers:
            return
        conn_ids = list(settings_watchers)
        disconnected = await event_emitter.emit(event_name, data, conn_ids=conn_ids)
        cleanup_disconnected_settings(disconnected)

def cleanup_disconnected_settings(disconnected_ids: list[str]):
    """Remove disconnected websockets from settings subscriptions"""
    for conn_id in disconnected_ids:
        settings_watchers.discard(conn_id)

def remove_id(conn_id: str):
    """Clean up subscriptions for a disconnected websocket ID"""
    cleanup_disconnected_settings([conn_id])