"""WebSocket manager for real-time communication."""
from fastapi import WebSocket
from typing import List, Dict, Any
import json
from loguru import logger


class WebSocketManager:
    """Manage WebSocket connections and broadcasting."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_detection(self, detection_data: Dict[str, Any]):
        """Broadcast detection event to all clients."""
        message = {
            'type': 'detection',
            'data': detection_data
        }
        await self.broadcast(message)
    
    async def broadcast_statistics(self, stats: Dict[str, Any]):
        """Broadcast statistics update to all clients."""
        message = {
            'type': 'statistics',
            'data': stats
        }
        await self.broadcast(message)
    
    async def broadcast_frame(self, frame_data: bytes):
        """Broadcast video frame to all clients."""
        # For video frames, send as binary data
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_bytes(frame_data)
            except Exception as e:
                logger.error(f"Error broadcasting frame: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)
