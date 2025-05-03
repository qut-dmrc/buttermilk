"""Activity tracker for monitoring system load and interactive user sessions.

This module provides functionality to track system activity like active 
websocket connections and API requests, which helps determine when 
the system is idle enough to process background batch jobs.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Set

# Global singleton instance
_instance = None


class ActivityTracker:
    """Track system activity to determine when it's safe to run background jobs.
    
    This class implements a central tracking point for:
    - Active websocket connections
    - Recent API requests
    - Any other activity indicators
    
    It provides methods to determine if the system is idle.
    """
    
    def __init__(self, idle_timeout_seconds: float = 60.0):
        """Initialize the activity tracker.
        
        Args:
            idle_timeout_seconds: Time in seconds after which the system is 
                                  considered idle if no activity is detected
        """
        self.idle_timeout = idle_timeout_seconds
        self.active_websockets: Set[str] = set()
        self.api_requests: List[datetime] = []
        self.last_activity_time = datetime.now()
        
    def register_websocket(self, connection_id: str) -> None:
        """Register an active websocket connection.
        
        Args:
            connection_id: Unique identifier for the websocket connection
        """
        self.active_websockets.add(connection_id)
        self.last_activity_time = datetime.now()
        
    def unregister_websocket(self, connection_id: str) -> None:
        """Unregister a websocket connection when it closes.
        
        Args:
            connection_id: Unique identifier for the websocket connection
        """
        if connection_id in self.active_websockets:
            self.active_websockets.remove(connection_id)
        self.last_activity_time = datetime.now()
        
    def record_api_request(self) -> None:
        """Record an API request.
        
        This should be called whenever an API endpoint is hit.
        """
        self.api_requests.append(datetime.now())
        self.last_activity_time = datetime.now()
        
        # Cleanup old API requests
        self._cleanup_old_api_requests()
        
    def _cleanup_old_api_requests(self) -> None:
        """Remove API requests older than the idle timeout."""
        cutoff_time = datetime.now() - timedelta(seconds=self.idle_timeout)
        self.api_requests = [t for t in self.api_requests if t >= cutoff_time]
        
    def is_idle(self) -> bool:
        """Check if the system is idle.
        
        Returns:
            True if the system is idle (no websockets or recent API requests),
            False otherwise
        """
        # Clean up old requests
        self._cleanup_old_api_requests()
        
        # System is idle if:
        # 1. No active websocket connections
        # 2. No recent API requests
        no_websockets = len(self.active_websockets) == 0
        no_recent_requests = len(self.api_requests) == 0
        
        # Also consider the time since any activity
        time_since_activity = datetime.now() - self.last_activity_time
        activity_timeout = time_since_activity.total_seconds() > self.idle_timeout
        
        return (no_websockets and no_recent_requests) or activity_timeout


def get_instance() -> ActivityTracker:
    """Get the global ActivityTracker instance.
    
    Returns:
        The global ActivityTracker instance
    """
    global _instance
    if _instance is None:
        _instance = ActivityTracker()
    return _instance
