"""
Simplified Health Monitoring for Basic Error Detection.

This module provides essential health monitoring focused on:
- Fatal error detection and system exit capability
- Basic flow responsiveness checks
- Simple health status reporting

External tools handle resource monitoring and process restarting.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from buttermilk._core import logger

from .health_monitor import HealthMonitor
from .metrics_collector import MetricsCollector, get_metrics_collector


class HealthStatus(Enum):
    """Basic health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SystemStatus:
    """Simple system status information."""
    overall_status: HealthStatus
    timestamp: datetime
    components: Dict[str, Any]
    error_message: Optional[str] = None


class SimpleHealthMonitor:
    """Simplified health monitoring for basic error detection and flow responsiveness."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 health_monitor: Optional[HealthMonitor] = None):
        """Initialize simplified health monitor."""
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.health_monitor = health_monitor or HealthMonitor()
        
        # Simple state tracking
        self.last_check: Optional[datetime] = None
        self.fatal_error_detected = False
        self.fatal_error_message: Optional[str] = None

    def report_fatal_error(self, error_message: str):
        """Report a fatal error that should cause system exit."""
        self.fatal_error_detected = True
        self.fatal_error_message = error_message
        logger.critical(f"Fatal error detected: {error_message}")

    def check_fatal_errors(self) -> bool:
        """Check if any fatal errors have been detected."""
        return self.fatal_error_detected

    def get_fatal_error_message(self) -> Optional[str]:
        """Get the fatal error message if any."""
        return self.fatal_error_message

    def check_flow_responsiveness(self, flow_name: str, timeout_seconds: int = 300) -> bool:
        """
        Check if a flow is responsive within timeout.
        
        Returns True if responsive, False if potentially stuck.
        This is a placeholder for future implementation.
        """
        # TODO: Implement actual flow responsiveness checking
        # For now, return True (no flows detected as stuck)
        logger.debug(f"Checking flow responsiveness: {flow_name} (timeout: {timeout_seconds}s)")
        return True

    def check_interactive_flow_ui_timeout(self, session_id: str, timeout_seconds: int = 1800) -> bool:
        """
        Check if interactive flow has been without UI for too long.
        
        Returns True if okay, False if stuck without UI.
        This is a placeholder for future implementation.
        """
        # TODO: Implement actual UI timeout checking
        # For now, return True (no flows detected as stuck without UI)
        logger.debug(f"Checking interactive flow UI timeout: {session_id} (timeout: {timeout_seconds}s)")
        return True

    def get_basic_health_status(self) -> SystemStatus:
        """Get basic system health status."""
        self.last_check = datetime.now()
        
        # Check for fatal errors first
        if self.fatal_error_detected:
            return SystemStatus(
                overall_status=HealthStatus.UNHEALTHY,
                timestamp=self.last_check,
                components={"fatal_error": {"detected": True, "message": self.fatal_error_message}},
                error_message=self.fatal_error_message
            )
        
        # Get basic health information
        try:
            health_summary = self.health_monitor.get_health_summary()
            overall_status_str = health_summary.get("overall_status", "unknown")
            
            # Convert to our simplified enum
            if overall_status_str == "healthy":
                overall_status = HealthStatus.HEALTHY
            elif overall_status_str == "degraded":
                overall_status = HealthStatus.DEGRADED
            elif overall_status_str == "unhealthy":
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN
            
            return SystemStatus(
                overall_status=overall_status,
                timestamp=self.last_check,
                components=health_summary.get("components", {}),
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return SystemStatus(
                overall_status=HealthStatus.UNKNOWN,
                timestamp=self.last_check,
                components={"error": str(e)},
                error_message=f"Health check failed: {e}"
            )

    def get_basic_metrics_summary(self) -> Dict[str, Any]:
        """Get basic metrics summary for simple monitoring."""
        try:
            flow_metrics = self.metrics_collector.get_flow_metrics()
            system_metrics = self.metrics_collector.get_system_metrics()
            
            return {
                "total_flows": len(flow_metrics),
                "system_uptime_seconds": (datetime.now() - system_metrics["start_time"]).total_seconds(),
                "memory_mb": system_metrics.get("total_memory_mb", 0),
                "active_sessions": system_metrics.get("active_sessions", 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global health monitor instance
_global_health_monitor: Optional[SimpleHealthMonitor] = None

def get_observability_manager() -> SimpleHealthMonitor:
    """Get or create global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = SimpleHealthMonitor()
    return _global_health_monitor

def get_simple_health_monitor() -> SimpleHealthMonitor:
    """Alias for get_observability_manager for consistency."""
    return get_observability_manager()

async def shutdown_observability_manager():
    """Shutdown global health monitor."""
    global _global_health_monitor
    if _global_health_monitor:
        # No complex shutdown needed for simplified version
        logger.info("Health monitor shutdown completed")
        _global_health_monitor = None
