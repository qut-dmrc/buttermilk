"""
Simplified Health Monitoring for Basic Status Checks.

This module provides essential health monitoring focused on:
- Basic system resource checks
- Simple component status tracking
- Fatal error detection

Complex thresholds, threading, and detailed monitoring are handled externally.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from buttermilk._core import logger


class HealthStatus(Enum):
    """System component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Simplified health status for individual system component."""
    component_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_health(self, status: HealthStatus, error_message: Optional[str] = None, **metadata):
        """Update component health status."""
        self.status = status
        self.last_check = datetime.now()
        self.error_message = error_message
        self.metadata.update(metadata)


class HealthMonitor:
    """Simplified health monitoring for basic status checks."""

    def __init__(self):
        """Initialize simplified health monitor."""
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # Initialize basic system health check
        self._register_basic_system_check()

    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for specific component."""
        return self.component_health.get(component_name)

    def get_all_health_status(self) -> Dict[str, ComponentHealth]:
        """Get health status for all components."""
        return dict(self.component_health)

    def get_overall_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.component_health:
            return HealthStatus.UNKNOWN

        statuses = [health.status for health in self.component_health.values()]

        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """Get basic health summary."""
        overall_status = self.get_overall_health_status()

        components_by_status = {
            "healthy": [],
            "degraded": [],
            "unhealthy": [],
            "unknown": []
        }

        for component_name, health in self.component_health.items():
            components_by_status[health.status.value].append({
                "name": component_name,
                "last_check": health.last_check.isoformat() if health.last_check else None,
                "error_message": health.error_message
            })

        return {
            "overall_status": overall_status.value,
            "components": components_by_status,
            "total_components": len(self.component_health),
            "last_check": datetime.now().isoformat()
        }

    def check_basic_system_health(self):
        """Perform basic system health check."""
        try:
            import psutil
            
            # Basic memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Basic CPU check
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Determine basic status
            status = HealthStatus.HEALTHY
            issues = []

            # Simple thresholds - external monitoring handles detailed alerting
            if memory_percent >= 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent >= 90:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory_percent:.1f}%")

            if cpu_percent >= 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent >= 90:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            # Update component health
            if "system_resources" not in self.component_health:
                self.component_health["system_resources"] = ComponentHealth(
                    component_name="system_resources"
                )

            self.component_health["system_resources"].update_health(
                status=status,
                error_message="; ".join(issues) if issues else None,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                memory_mb=memory.used / 1024 / 1024
            )

        except Exception as e:
            if "system_resources" not in self.component_health:
                self.component_health["system_resources"] = ComponentHealth(
                    component_name="system_resources"
                )
            
            self.component_health["system_resources"].update_health(
                status=HealthStatus.UNKNOWN,
                error_message=f"Failed to check system resources: {e}"
            )
            logger.error(f"System health check failed: {e}")

    def _register_basic_system_check(self):
        """Register basic system health component."""
        self.component_health["system_resources"] = ComponentHealth(
            component_name="system_resources"
        )
        # Perform initial check
        self.check_basic_system_health()


# Removed complex health check factory functions - these were over-engineered
# External monitoring tools should handle flow and agent specific monitoring
