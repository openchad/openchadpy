"""
Event Emitter for WebSocket and Tauri-based event broadcasting.
This module provides a robust event emitter that allows Python tools to send
events to connected frontend clients via WebSocket (browser mode) or 
pytauri Emitter (Tauri desktop mode).
"""
import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)
# Tauri event names only allow: alphanumeric, '-', '/', ':', '_'
# Dots (in table names like "db.table") and backslashes (Windows paths) are illegal.
_TAURI_EVENT_SAFE = re.compile(r'[^a-zA-Z0-9\-/:_]')

def sanitize_tauri_event(event: str) -> str:
    """
    Replace any character not allowed in Tauri event names with '_'.
    Converts backslashes to '/' first (Windows paths), then replaces dots and others.
    """
    event = event.replace('\\', '/')   # \ → / (forward slash is allowed)
    return _TAURI_EVENT_SAFE.sub('_', event)
# Optional pytauri imports - only available in Tauri mode
_app_handle = None
_emitter_available = False
try:
    from pytauri.ffi import Emitter, AppHandle
    _emitter_available = True
except ImportError:
    pass

def set_app_handle(app_handle: Any) -> None:
    """
    Set the Tauri AppHandle for event emission in desktop mode.
    Called during Tauri app initialization.
    """
    global _app_handle
    _app_handle = app_handle
    logger.info("Tauri AppHandle set for event emission")

def is_tauri_mode() -> bool:
    """Check if running in Tauri mode with AppHandle available."""
    return _emitter_available and _app_handle is not None

@dataclass
class EventSubscription:
    """Represents a subscription to a specific event."""
    event_name: str
    conn_ids: Set[str] = field(default_factory=set)

class EventEmitter:
    """
    Event emitter for broadcasting events to frontend clients.
    Supports both WebSocket (browser) and pytauri (Tauri desktop) modes.
    Supports:
    - Broadcasting to all connected clients
    - Broadcasting to specific connection IDs (WebSocket mode)
    - Event-based subscriptions (clients can subscribe to specific events)
    - Fire-and-forget async event emission
    Usage:
        emitter = EventEmitter.get_instance()
        # Send event to all clients
        await emitter.emit("task:progress", {"progress": 50})
        # Send event to specific clients (WebSocket mode only)
        await emitter.emit("task:complete", {"result": "done"}, conn_ids=["abc-123"])
        # Fire and forget (non-blocking)
        emitter.emit_nowait("notification", {"message": "Hello"})
    """
    _instance: Optional["EventEmitter"] = None

    def __init__(self):
        # Event subscriptions: event_name -> set of connection IDs
        self._subscriptions: Dict[str, Set[str]] = {}
        # Event history for replay (optional, limited size)
        self._event_history: List[Dict[str, Any]] = []
        self._max_history_size = 100
        # WebSocket manager (lazy import to avoid circular deps)
        self._manager = None
    @classmethod

    def get_instance(cls) -> "EventEmitter":
        """Get the singleton instance of EventEmitter."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_ws_manager(self):
        """Lazy import of connection manager to avoid circular deps."""
        if self._manager is None:
            try:
                from .connection_manager import manager
                self._manager = manager
            except ImportError:
                # Fallback for non-package execution context
                from openchadpy.connection_manager import manager
                self._manager = manager
        return self._manager

    async def emit(
        self,
        event: str,
        data: Optional[Dict[str, Any]] = None,
        conn_ids: Optional[List[str]] = None,
        exclude_conn_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Emit an event to frontend clients.
        In Tauri mode: Uses pytauri Emitter to emit to the webview.
        In WebSocket mode: Broadcasts to connected WebSocket clients.
        Args:
            event: The event name (e.g., "task:progress", "notification")
            data: Optional data payload to send with the event
            conn_ids: Optional list of specific connection IDs to send to (WebSocket only).
                     If None, broadcasts to all subscribed clients for this event,
                     or all connected clients if no subscriptions exist.
            exclude_conn_ids: Optional list of connection IDs to exclude (WebSocket only)
        Returns:
            List of connection IDs that failed to receive the event (WebSocket mode)
            Empty list in Tauri mode
        """
        payload = data or {}
        manager = self._get_ws_manager()
        # Tauri mode: Use pytauri Emitter
        if is_tauri_mode():
            try:
                from pytauri.ffi import Emitter
                safe_event = sanitize_tauri_event(event)
                json_payload = json.dumps(payload)
                Emitter.emit_str(_app_handle, safe_event, json_payload) #pyrefly: ignore
                logger.debug(f"Emitted event '{safe_event}' via Tauri Emitter")
                message = {
                    "event": event,
                    "response": payload
                }
                try:
                    self._add_to_history(event, payload, ["tauri"])
                    await manager.broadcast(message, manager.active_websockets.keys()) #pyrefly: ignore
                except Exception as e:
                    pass
                return []
            except Exception as e:
                logger.error(f"Failed to emit event via Tauri: {e}")
                return []
        message = {
            "event": event,
            "response": payload
        }
        # Determine target connections
        if conn_ids is not None:
            targets = set(conn_ids)
        elif event in self._subscriptions and self._subscriptions[event]:
            # Use subscribers for this event
            targets = self._subscriptions[event].copy()
        else:
            # Broadcast to all connected clients
            targets = set(manager.active_websockets.keys())
        # Apply exclusions
        if exclude_conn_ids:
            targets -= set(exclude_conn_ids)
        if not targets:
            return []
        # Broadcast the event
        failed = await manager.broadcast(message, list(targets))
        # Clean up failed connections from subscriptions
        for failed_id in failed:
            self._remove_connection(failed_id)
        # Store in history (optional)
        self._add_to_history(event, payload, list(targets))
        logger.debug(f"Emitted event '{event}' to {len(targets)} clients, {len(failed)} failed")
        return failed

    def emit_nowait(
        self,
        event: str,
        data: Optional[Dict[str, Any]] = None,
        conn_ids: Optional[List[str]] = None,
        exclude_conn_ids: Optional[List[str]] = None
    ) -> None:
        """
        Fire-and-forget event emission.
        Same as emit() but doesn't wait for completion.
        Useful for non-critical events where you don't need confirmation.
        """
        asyncio.create_task(
            self.emit(event, data, conn_ids, exclude_conn_ids)
        )

    def emit_sync(
        self,
        event: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Synchronous event emission (Tauri mode only).
        Falls back to emit_nowait in WebSocket mode.
        """
        payload = data or {}
        if is_tauri_mode():
            try:
                from pytauri.ffi import Emitter
                safe_event = sanitize_tauri_event(event)
                json_payload = json.dumps(payload)
                Emitter.emit_str(_app_handle, safe_event, json_payload) #pyrefly: ignore
                logger.debug(f"Emitted event '{safe_event}' via Tauri Emitter (sync)")
            except Exception as e:
                logger.error(f"Failed to emit event via Tauri: {e}")
        else:
            # Fallback to async in WebSocket mode
            self.emit_nowait(event, data)

    def subscribe(self, event: str, conn_id: str) -> None:
        """
        Subscribe a connection to a specific event.
        Args:
            event: The event name to subscribe to
            conn_id: The connection ID to subscribe
        """
        if event not in self._subscriptions:
            self._subscriptions[event] = set()
        self._subscriptions[event].add(conn_id)
        logger.debug(f"Connection {conn_id} subscribed to event '{event}'")

    def unsubscribe(self, event: str, conn_id: str) -> bool:
        """
        Unsubscribe a connection from a specific event.
        Args:
            event: The event name to unsubscribe from
            conn_id: The connection ID to unsubscribe
        Returns:
            True if the connection was subscribed, False otherwise
        """
        if event in self._subscriptions:
            if conn_id in self._subscriptions[event]:
                self._subscriptions[event].discard(conn_id)
                logger.debug(f"Connection {conn_id} unsubscribed from event '{event}'")
                return True
        return False

    def unsubscribe_all(self, conn_id: str) -> None:
        """
        Unsubscribe a connection from all events.
        Useful when a connection disconnects.
        Args:
            conn_id: The connection ID to unsubscribe
        """
        self._remove_connection(conn_id)
 
    def get_subscribers(self, event: str) -> List[str]:
        """
        Get all connection IDs subscribed to an event.
        Args:
            event: The event name
        Returns:
            List of subscribed connection IDs
        """
        if event in self._subscriptions:
            return list(self._subscriptions[event])
        return []
 
    def _remove_connection(self, conn_id: str) -> None:
        """Remove a connection from all subscriptions."""
        for subscribers in self._subscriptions.values():
            subscribers.discard(conn_id)
 
    def _add_to_history(
        self,
        event: str,
        data: Optional[Dict[str, Any]],
        targets: List[str]
    ) -> None:
        """Add an event to the history buffer."""
        import time
        self._event_history.append({
            "event": event,
            "data": data,
            "targets": len(targets),
            "timestamp": time.time()
        })
        # Trim history if needed
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
 
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent event history."""
        return self._event_history[-limit:]
 
    def create_send_event(self, conn_id: Optional[str] = None) -> Callable[[str, Dict[str, Any]], None]:
        """
        Create a send_event function for tools.
        This creates a fire-and-forget function that tools can use to send events.
        Works in both WebSocket and Tauri modes.
        Args:
            conn_id: Optional connection ID to scope events to a specific client
                    (WebSocket mode only, ignored in Tauri mode)
        Returns:
            A callable that accepts (event_name, data) and emits the event
        """
        if conn_id and not is_tauri_mode():
            def send_event_scoped(event: str, data: Dict[str, Any]) -> None:
                self.emit_nowait(event, data, conn_ids=[conn_id])
            return send_event_scoped
        else:
            def send_event_broadcast(event: str, data: Dict[str, Any]) -> None:
                if is_tauri_mode():
                    self.emit_sync(event, data)
                else:
                    self.emit_nowait(event, data)
            return send_event_broadcast
        
# Global singleton instance
event_emitter = EventEmitter.get_instance()