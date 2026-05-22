import uuid
from fastapi import WebSocket
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)
class ConnectionManager:
    """Manages active WebSocket connections."""
    
    def __init__(self):
        # Active websockets by uuid
        self.active_websockets: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept connection and register websocket. Returns a unique connection ID."""
        await websocket.accept()
        conn_id = str(uuid.uuid4())
        self.active_websockets[conn_id] = websocket
        logger.info(f"WebSocket connected: {conn_id}")
        return conn_id
    
    def disconnect(self, conn_id: str):
        """Unregister websocket."""
        if conn_id in self.active_websockets:
            del self.active_websockets[conn_id]
            logger.info(f"WebSocket disconnected: {conn_id}")
    
    def get_socket(self, conn_id: str) -> WebSocket | None:
        """Get websocket by ID."""
        return self.active_websockets.get(conn_id)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific websocket."""
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to client (likely disconnected): {e}")
            return False

    async def broadcast(self, message: dict, conn_ids: List[str]) -> List[str]:
        """
        Broadcast message to a list of websocket IDs.
        Returns a list of IDs that failed (likely disconnected).
        """
        disconnected = []
        for cid in conn_ids:
            ws = self.get_socket(cid)
            if ws:
                try:
                    logger.info(f"Broadcasting to client {cid}")
                    await ws.send_json(message)
                except (Exception, RuntimeError) as e:
                    logger.debug(f"Broadcast failed for {cid}: {e}")
                    disconnected.append(cid)
            else:
                # Socket is gone from manager but still in some subscription list
                disconnected.append(cid)
        # Clean up local references if failed
        for cid in disconnected:
            if cid in self.active_websockets:
                del self.active_websockets[cid]
        return disconnected
# Global instance
manager = ConnectionManager()
