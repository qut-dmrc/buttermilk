"""WebSocket connection tracking for the ActivityTracker.

This module provides utilities to track WebSocket connections and
register them with the ActivityTracker to help determine system idle status.
"""

import uuid
from typing import Dict, Set

from starlette.websockets import WebSocket, WebSocketDisconnect

from buttermilk.bm import logger
from buttermilk.web.activity_tracker import get_instance as get_activity_tracker


class WebSocketTracker:
    """Tracks active WebSocket connections for the ActivityTracker."""
    
    def __init__(self):
        """Initialize the WebSocket tracker."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_ids: Set[str] = set()
        self.activity_tracker = get_activity_tracker()
    
    async def connect(self, websocket: WebSocket) -> str:
        """Register a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to register
            
        Returns:
            Connection ID that can be used to unregister later
        """
        # Accept the connection
        await websocket.accept()
        
        # Generate a unique ID for this connection
        connection_id = str(uuid.uuid4())
        
        # Store the connection
        self.active_connections[connection_id] = websocket
        self.connection_ids.add(connection_id)
        
        # Register with activity tracker
        self.activity_tracker.register_websocket(connection_id)
        
        logger.debug(f"WebSocket connection {connection_id} registered")
        return connection_id
    
    def disconnect(self, connection_id: str) -> None:
        """Unregister a WebSocket connection.
        
        Args:
            connection_id: The ID of the connection to unregister
        """
        if connection_id in self.active_connections:
            self.active_connections.pop(connection_id)
            self.connection_ids.remove(connection_id)
            
            # Unregister from activity tracker
            self.activity_tracker.unregister_websocket(connection_id)
            
            logger.debug(f"WebSocket connection {connection_id} unregistered")
    
    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all active connections.
        
        Args:
            message: The message to broadcast
        """
        disconnected = []
        
        # Try to send to each connection
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                # Connection closed, mark for removal
                disconnected.append(connection_id)
            except Exception as e:
                logger.error(f"Error sending to WebSocket {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)


# Global singleton instance
_instance = None


def get_instance() -> WebSocketTracker:
    """Get the global WebSocketTracker instance.
    
    Returns:
        The global WebSocketTracker instance
    """
    global _instance
    if _instance is None:
        _instance = WebSocketTracker()
    return _instance
