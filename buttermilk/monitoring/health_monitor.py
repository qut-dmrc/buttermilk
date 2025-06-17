"""
Production Health Monitoring for Flow-Agnostic System Health.

This module provides comprehensive health monitoring for all system components,
including database connections, vector stores, agent health, and system resources.
"""

import asyncio
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Awaitable
from enum import Enum

from buttermilk._core import logger


class HealthStatus(Enum):
    """System component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for individual system component."""
    component_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    check_count: int = 0
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    
    def update_health(self, status: HealthStatus, response_time_ms: Optional[float] = None,
                     error_message: Optional[str] = None, **metadata):
        """Update component health status."""
        previous_status = self.status
        self.status = status
        self.last_check = datetime.now()
        self.response_time_ms = response_time_ms
        self.error_message = error_message
        self.metadata.update(metadata)
        self.check_count += 1
        
        if status == HealthStatus.HEALTHY:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        # Update uptime percentage (simple sliding window)
        if self.check_count > 0:
            failures = self.consecutive_failures if self.consecutive_failures < self.check_count else self.check_count
            self.uptime_percentage = ((self.check_count - failures) / self.check_count) * 100


class HealthMonitor:
    """Comprehensive health monitoring for production systems."""
    
    def __init__(self, check_interval_seconds: int = 30):
        """Initialize health monitor with configurable check interval."""
        self.check_interval = check_interval_seconds
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}
        self.is_monitoring = False
        self.monitor_task = None
        
        # System thresholds
        self.thresholds = {
            "memory_percent_warning": 85.0,
            "memory_percent_critical": 95.0,
            "cpu_percent_warning": 80.0,
            "cpu_percent_critical": 95.0,
            "disk_percent_warning": 85.0,
            "disk_percent_critical": 95.0,
            "response_time_warning_ms": 1000.0,
            "response_time_critical_ms": 5000.0
        }
        
        # Register default system checks
        self._register_default_checks()
    
    def register_health_check(self, component_name: str, 
                            health_check_func: Callable[[], Awaitable[ComponentHealth]]):
        """Register a custom health check function for a component."""
        self.health_checks[component_name] = health_check_func
        self.component_health[component_name] = ComponentHealth(component_name=component_name)
        logger.info(f"Registered health check for component: {component_name}")
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started health monitoring with {self.check_interval}s interval")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health monitoring")
    
    async def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Run health checks for all registered components."""
        results = {}
        
        for component_name, health_check_func in self.health_checks.items():
            try:
                start_time = time.time()
                component_health = await health_check_func()
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update response time if not set by health check
                if component_health.response_time_ms is None:
                    component_health.response_time_ms = response_time
                
                self.component_health[component_name] = component_health
                results[component_name] = component_health
                
            except Exception as e:
                # Health check failed
                error_health = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
                self.component_health[component_name] = error_health
                results[component_name] = error_health
                logger.error(f"Health check failed for {component_name}: {e}")
        
        return results
    
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
        """Get comprehensive health summary."""
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
                "response_time_ms": health.response_time_ms,
                "uptime_percentage": health.uptime_percentage,
                "consecutive_failures": health.consecutive_failures,
                "error_message": health.error_message
            })
        
        return {
            "overall_status": overall_status.value,
            "components": components_by_status,
            "total_components": len(self.component_health),
            "last_check": datetime.now().isoformat(),
            "monitoring_active": self.is_monitoring
        }
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self.check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        async def check_system_resources():
            """Check system resource utilization."""
            try:
                # Memory check
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # CPU check
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Disk check
                disk = psutil.disk_usage('/')
                disk_percent = disk.used / disk.total * 100
                
                # Determine status based on thresholds
                status = HealthStatus.HEALTHY
                issues = []
                
                if memory_percent >= self.thresholds["memory_percent_critical"]:
                    status = HealthStatus.UNHEALTHY
                    issues.append(f"Critical memory usage: {memory_percent:.1f}%")
                elif memory_percent >= self.thresholds["memory_percent_warning"]:
                    status = HealthStatus.DEGRADED
                    issues.append(f"High memory usage: {memory_percent:.1f}%")
                
                if cpu_percent >= self.thresholds["cpu_percent_critical"]:
                    status = HealthStatus.UNHEALTHY
                    issues.append(f"Critical CPU usage: {cpu_percent:.1f}%")
                elif cpu_percent >= self.thresholds["cpu_percent_warning"]:
                    status = HealthStatus.DEGRADED
                    issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                
                if disk_percent >= self.thresholds["disk_percent_critical"]:
                    status = HealthStatus.UNHEALTHY
                    issues.append(f"Critical disk usage: {disk_percent:.1f}%")
                elif disk_percent >= self.thresholds["disk_percent_warning"]:
                    status = HealthStatus.DEGRADED
                    issues.append(f"High disk usage: {disk_percent:.1f}%")
                
                return ComponentHealth(
                    component_name="system_resources",
                    status=status,
                    error_message="; ".join(issues) if issues else None,
                    memory_percent=memory_percent,
                    cpu_percent=cpu_percent,
                    disk_percent=disk_percent,
                    memory_mb=memory.used / 1024 / 1024,
                    available_memory_mb=memory.available / 1024 / 1024
                )
                
            except Exception as e:
                return ComponentHealth(
                    component_name="system_resources",
                    status=HealthStatus.UNHEALTHY,
                    error_message=f"Failed to check system resources: {e}"
                )
        
        async def check_database_connection():
            """Check database connectivity (placeholder - would integrate with actual DB)."""
            try:
                # Simulate database health check
                start_time = time.time()
                await asyncio.sleep(0.01)  # Simulate DB query
                response_time = (time.time() - start_time) * 1000
                
                status = HealthStatus.HEALTHY
                if response_time > self.thresholds["response_time_critical_ms"]:
                    status = HealthStatus.UNHEALTHY
                elif response_time > self.thresholds["response_time_warning_ms"]:
                    status = HealthStatus.DEGRADED
                
                return ComponentHealth(
                    component_name="database",
                    status=status,
                    response_time_ms=response_time,
                    connection_pool_size=10,  # Mock data
                    active_connections=3
                )
                
            except Exception as e:
                return ComponentHealth(
                    component_name="database",
                    status=HealthStatus.UNHEALTHY,
                    error_message=f"Database connection failed: {e}"
                )
        
        async def check_vector_store():
            """Check vector store connectivity and performance."""
            try:
                # Simulate vector store health check
                start_time = time.time()
                await asyncio.sleep(0.02)  # Simulate vector query
                response_time = (time.time() - start_time) * 1000
                
                status = HealthStatus.HEALTHY
                if response_time > self.thresholds["response_time_critical_ms"]:
                    status = HealthStatus.UNHEALTHY
                elif response_time > self.thresholds["response_time_warning_ms"]:
                    status = HealthStatus.DEGRADED
                
                return ComponentHealth(
                    component_name="vector_store",
                    status=status,
                    response_time_ms=response_time,
                    index_size=50000,  # Mock data
                    last_update="2025-01-17T10:30:00Z"
                )
                
            except Exception as e:
                return ComponentHealth(
                    component_name="vector_store",
                    status=HealthStatus.UNHEALTHY,
                    error_message=f"Vector store check failed: {e}"
                )
        
        async def check_websocket_health():
            """Check WebSocket infrastructure health."""
            try:
                # Simulate WebSocket health check
                from buttermilk.monitoring.metrics_collector import get_metrics_collector
                metrics = get_metrics_collector()
                system_metrics = metrics.get_system_metrics()
                
                websocket_connections = system_metrics.get("websocket_connections", 0)
                active_sessions = system_metrics.get("active_sessions", 0)
                
                # Determine status based on load
                status = HealthStatus.HEALTHY
                if websocket_connections > 100:  # Threshold for high load
                    status = HealthStatus.DEGRADED
                elif websocket_connections > 200:  # Critical threshold
                    status = HealthStatus.UNHEALTHY
                
                return ComponentHealth(
                    component_name="websocket_infrastructure",
                    status=status,
                    active_connections=websocket_connections,
                    active_sessions=active_sessions,
                    max_connections=250  # Mock limit
                )
                
            except Exception as e:
                return ComponentHealth(
                    component_name="websocket_infrastructure",
                    status=HealthStatus.UNHEALTHY,
                    error_message=f"WebSocket health check failed: {e}"
                )
        
        # Register default checks
        self.register_health_check("system_resources", check_system_resources)
        self.register_health_check("database", check_database_connection)
        self.register_health_check("vector_store", check_vector_store)
        self.register_health_check("websocket_infrastructure", check_websocket_health)


def create_flow_health_check(flow_name: str) -> Callable[[], Awaitable[ComponentHealth]]:
    """Create a health check function for a specific flow."""
    
    async def check_flow_health():
        """Check health of specific flow."""
        try:
            from buttermilk.monitoring.metrics_collector import get_metrics_collector
            metrics = get_metrics_collector()
            flow_metrics = metrics.get_flow_metrics(flow_name)
            
            if flow_name not in flow_metrics:
                return ComponentHealth(
                    component_name=f"flow_{flow_name}",
                    status=HealthStatus.UNKNOWN,
                    error_message="No metrics available for flow"
                )
            
            flow_data = flow_metrics[flow_name]
            
            # Determine health based on error rate and performance
            status = HealthStatus.HEALTHY
            issues = []
            
            if flow_data.error_rate > 0.2:  # 20% error rate
                status = HealthStatus.UNHEALTHY
                issues.append(f"High error rate: {flow_data.error_rate:.1%}")
            elif flow_data.error_rate > 0.1:  # 10% error rate
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated error rate: {flow_data.error_rate:.1%}")
            
            if flow_data.p95_execution_time > 10000:  # 10 seconds
                status = HealthStatus.UNHEALTHY
                issues.append(f"Slow P95 execution time: {flow_data.p95_execution_time:.0f}ms")
            elif flow_data.p95_execution_time > 5000:  # 5 seconds
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated P95 execution time: {flow_data.p95_execution_time:.0f}ms")
            
            return ComponentHealth(
                component_name=f"flow_{flow_name}",
                status=status,
                error_message="; ".join(issues) if issues else None,
                total_executions=flow_data.total_executions,
                error_rate=flow_data.error_rate,
                avg_execution_time=flow_data.avg_execution_time,
                p95_execution_time=flow_data.p95_execution_time
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name=f"flow_{flow_name}",
                status=HealthStatus.UNHEALTHY,
                error_message=f"Flow health check failed: {e}"
            )
    
    return check_flow_health


def create_agent_health_check(agent_name: str, flow_name: str) -> Callable[[], Awaitable[ComponentHealth]]:
    """Create a health check function for a specific agent."""
    
    async def check_agent_health():
        """Check health of specific agent."""
        try:
            from buttermilk.monitoring.metrics_collector import get_metrics_collector
            metrics = get_metrics_collector()
            agent_metrics = metrics.get_agent_metrics(flow_name)
            
            agent_key = f"{flow_name}.{agent_name}"
            if agent_key not in agent_metrics:
                return ComponentHealth(
                    component_name=f"agent_{agent_key}",
                    status=HealthStatus.UNKNOWN,
                    error_message="No metrics available for agent"
                )
            
            agent_data = agent_metrics[agent_key]
            
            # Determine health based on response time and success rate
            status = HealthStatus.HEALTHY
            issues = []
            
            success_rate = (agent_data.successful_invocations / agent_data.total_invocations 
                           if agent_data.total_invocations > 0 else 0.0)
            
            if success_rate < 0.8:  # 80% success rate threshold
                status = HealthStatus.UNHEALTHY
                issues.append(f"Low success rate: {success_rate:.1%}")
            elif success_rate < 0.9:  # 90% success rate threshold
                status = HealthStatus.DEGRADED
                issues.append(f"Reduced success rate: {success_rate:.1%}")
            
            if agent_data.avg_response_time > 5000:  # 5 seconds
                status = HealthStatus.UNHEALTHY
                issues.append(f"Slow response time: {agent_data.avg_response_time:.0f}ms")
            elif agent_data.avg_response_time > 2000:  # 2 seconds
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated response time: {agent_data.avg_response_time:.0f}ms")
            
            return ComponentHealth(
                component_name=f"agent_{agent_key}",
                status=status,
                error_message="; ".join(issues) if issues else None,
                total_invocations=agent_data.total_invocations,
                success_rate=success_rate,
                avg_response_time=agent_data.avg_response_time,
                avg_confidence_score=agent_data.avg_confidence_score
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name=f"agent_{agent_key}",
                status=HealthStatus.UNHEALTHY,
                error_message=f"Agent health check failed: {e}"
            )
    
    return check_agent_health