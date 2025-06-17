"""
Production Observability Management for Real-time Monitoring and Alerting.

This module provides comprehensive observability features including:
- Real-time alerting based on metrics and health status
- Integration with monitoring systems (Prometheus, Grafana, etc.)
- Alert routing and notification management
- Performance dashboard data aggregation
- Incident tracking and escalation
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Union
from enum import Enum
import threading

from buttermilk._core import logger
from .metrics_collector import MetricsCollector, get_metrics_collector
from .health_monitor import HealthMonitor, HealthStatus


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Alert definition with context and routing information."""
    alert_id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    escalated: bool = False
    
    def acknowledge(self, acknowledged_by: str):
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = acknowledged_by
        logger.info(f"Alert {self.alert_id} acknowledged by {acknowledged_by}")
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
        logger.info(f"Alert {self.alert_id} resolved")


class AlertRule:
    """Alert rule definition for automated alerting."""
    
    def __init__(self, rule_id: str, condition_func: Callable[[Dict[str, Any]], bool],
                 alert_level: AlertLevel, message_template: str,
                 component: str, cooldown_seconds: int = 300):
        self.rule_id = rule_id
        self.condition_func = condition_func
        self.alert_level = alert_level
        self.message_template = message_template
        self.component = component
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered = None
        self.enabled = True
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Check if alert rule should trigger based on current metrics."""
        if not self.enabled:
            return False
        
        # Check cooldown period
        if (self.last_triggered and 
            datetime.now() - self.last_triggered < timedelta(seconds=self.cooldown_seconds)):
            return False
        
        return self.condition_func(metrics)
    
    def trigger(self, metrics: Dict[str, Any]) -> Alert:
        """Trigger alert and return Alert object."""
        self.last_triggered = datetime.now()
        
        alert_id = f"{self.rule_id}_{int(time.time())}"
        message = self.message_template.format(**metrics)
        
        return Alert(
            alert_id=alert_id,
            level=self.alert_level,
            component=self.component,
            message=message,
            timestamp=datetime.now(),
            metadata={
                "rule_id": self.rule_id,
                "metrics_snapshot": metrics
            }
        )


class ObservabilityManager:
    """Central observability management for production monitoring."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 health_monitor: Optional[HealthMonitor] = None):
        """Initialize observability manager."""
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.health_monitor = health_monitor or HealthMonitor()
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_handlers: List[Callable[[Alert], Any]] = []
        
        # Performance tracking
        self.performance_snapshots: List[Dict[str, Any]] = []
        self.dashboard_data: Dict[str, Any] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task = None
        self.snapshot_interval = 60  # seconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register default alert rules
        self._register_default_alert_rules()
    
    def register_alert_rule(self, rule: AlertRule):
        """Register a new alert rule."""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Registered alert rule: {rule.rule_id}")
    
    def register_alert_handler(self, handler: Callable[[Alert], Any]):
        """Register an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info("Registered alert handler")
    
    async def start_monitoring(self):
        """Start continuous observability monitoring."""
        if self.is_monitoring:
            logger.warning("Observability monitoring already started")
            return
        
        self.is_monitoring = True
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        # Start observability monitoring loop
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Started observability monitoring")
    
    async def stop_monitoring(self):
        """Stop observability monitoring."""
        self.is_monitoring = False
        
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
        
        # Stop monitoring loop
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped observability monitoring")
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by level."""
        with self._lock:
            alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
            
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        with self._lock:
            active_alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
            
            summary = {
                "total_active": len(active_alerts),
                "by_level": {level.value: 0 for level in AlertLevel},
                "unacknowledged": 0,
                "escalated": 0,
                "oldest_alert": None
            }
            
            for alert in active_alerts:
                summary["by_level"][alert.level.value] += 1
                if not alert.acknowledged:
                    summary["unacknowledged"] += 1
                if alert.escalated:
                    summary["escalated"] += 1
            
            if active_alerts:
                oldest = min(active_alerts, key=lambda a: a.timestamp)
                summary["oldest_alert"] = {
                    "alert_id": oldest.alert_id,
                    "timestamp": oldest.timestamp.isoformat(),
                    "component": oldest.component,
                    "level": oldest.level.value
                }
            
            return summary
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledge(acknowledged_by)
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolve()
                return True
            return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get aggregated data for monitoring dashboard."""
        with self._lock:
            # Get current metrics
            system_metrics = self.metrics_collector.get_system_metrics()
            flow_metrics = self.metrics_collector.get_flow_metrics()
            session_metrics = self.metrics_collector.get_session_metrics(active_only=True)
            health_summary = self.health_monitor.get_health_summary()
            alert_summary = self.get_alert_summary()
            
            # Calculate additional dashboard metrics
            total_flows = len(flow_metrics)
            total_executions = sum(m.total_executions for m in flow_metrics.values())
            avg_success_rate = (sum(m.successful_executions / m.total_executions 
                                  if m.total_executions > 0 else 0.0 
                                  for m in flow_metrics.values()) / total_flows) if total_flows > 0 else 0.0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "uptime_seconds": (datetime.now() - system_metrics["start_time"]).total_seconds(),
                    "memory_mb": system_metrics["total_memory_mb"],
                    "peak_memory_mb": system_metrics["peak_memory_mb"],
                    "cpu_usage_percent": system_metrics["cpu_usage_percent"],
                    "websocket_connections": system_metrics["websocket_connections"]
                },
                "flows": {
                    "total_flows": total_flows,
                    "total_executions": total_executions,
                    "avg_success_rate": avg_success_rate,
                    "active_sessions": len(session_metrics)
                },
                "health": health_summary,
                "alerts": alert_summary,
                "performance_trends": self._get_performance_trends()
            }
    
    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # System metrics
        system_metrics = self.metrics_collector.get_system_metrics()
        lines.append(f"buttermilk_memory_mb {system_metrics['total_memory_mb']}")
        lines.append(f"buttermilk_peak_memory_mb {system_metrics['peak_memory_mb']}")
        lines.append(f"buttermilk_cpu_percent {system_metrics['cpu_usage_percent']}")
        lines.append(f"buttermilk_websocket_connections {system_metrics['websocket_connections']}")
        lines.append(f"buttermilk_active_sessions {system_metrics['active_sessions']}")
        
        # Flow metrics
        flow_metrics = self.metrics_collector.get_flow_metrics()
        for flow_name, metrics in flow_metrics.items():
            labels = f'{{flow="{flow_name}"}}'
            lines.append(f"buttermilk_flow_executions_total{labels} {metrics.total_executions}")
            lines.append(f"buttermilk_flow_execution_time_avg{labels} {metrics.avg_execution_time}")
            lines.append(f"buttermilk_flow_error_rate{labels} {metrics.error_rate}")
        
        # Agent metrics
        agent_metrics = self.metrics_collector.get_agent_metrics()
        for agent_key, metrics in agent_metrics.items():
            labels = f'{{agent="{metrics.agent_name}",flow="{metrics.flow_name}"}}'
            lines.append(f"buttermilk_agent_invocations_total{labels} {metrics.total_invocations}")
            lines.append(f"buttermilk_agent_response_time_avg{labels} {metrics.avg_response_time}")
            lines.append(f"buttermilk_agent_confidence_avg{labels} {metrics.avg_confidence_score}")
        
        # Health status
        health_summary = self.health_monitor.get_health_summary()
        overall_health = 1 if health_summary["overall_status"] == "healthy" else 0
        lines.append(f"buttermilk_health_status {overall_health}")
        
        # Alert metrics
        active_alerts = self.get_active_alerts()
        lines.append(f"buttermilk_active_alerts_total {len(active_alerts)}")
        
        for level in AlertLevel:
            count = len([a for a in active_alerts if a.level == level])
            lines.append(f"buttermilk_alerts_by_level{{level=\"{level.value}\"}} {count}")
        
        return "\n".join(lines)
    
    async def _monitoring_loop(self):
        """Main observability monitoring loop."""
        while self.is_monitoring:
            try:
                # Take performance snapshot
                await self._take_performance_snapshot()
                
                # Check alert rules
                await self._check_alert_rules()
                
                # Auto-resolve alerts if conditions are met
                await self._auto_resolve_alerts()
                
                # Update dashboard data
                self.dashboard_data = self.get_dashboard_data()
                
                await asyncio.sleep(self.snapshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in observability monitoring loop: {e}")
                await asyncio.sleep(self.snapshot_interval)
    
    async def _take_performance_snapshot(self):
        """Take a performance snapshot for trending."""
        snapshot = {
            "timestamp": datetime.now(),
            "system_metrics": self.metrics_collector.get_system_metrics(),
            "health_summary": self.health_monitor.get_health_summary(),
            "active_alerts_count": len(self.get_active_alerts())
        }
        
        with self._lock:
            self.performance_snapshots.append(snapshot)
            
            # Keep only last 24 hours of snapshots
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_snapshots = [
                s for s in self.performance_snapshots 
                if s["timestamp"] >= cutoff_time
            ]
    
    async def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed."""
        # Get current metrics for rule evaluation
        dashboard_data = self.get_dashboard_data()
        
        for rule_id, rule in self.alert_rules.items():
            try:
                if rule.should_trigger(dashboard_data):
                    alert = rule.trigger(dashboard_data)
                    await self._handle_new_alert(alert)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_id}: {e}")
    
    async def _handle_new_alert(self, alert: Alert):
        """Handle a newly triggered alert."""
        with self._lock:
            self.active_alerts[alert.alert_id] = alert
        
        logger.warning(f"New {alert.level.value} alert: {alert.message}")
        
        # Call registered alert handlers
        for handler in self.alert_handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(None, handler, alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts that no longer meet trigger conditions."""
        # This is a simplified implementation - in production you'd have
        # specific resolution conditions for each alert type
        with self._lock:
            for alert in self.active_alerts.values():
                if not alert.resolved and alert.level == AlertLevel.INFO:
                    # Auto-resolve info alerts after 1 hour
                    if datetime.now() - alert.timestamp > timedelta(hours=1):
                        alert.resolve()
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends from snapshots."""
        if len(self.performance_snapshots) < 2:
            return {}
        
        recent_snapshots = self.performance_snapshots[-10:]  # Last 10 snapshots
        
        # Calculate memory trend
        memory_values = [s["system_metrics"]["total_memory_mb"] for s in recent_snapshots]
        memory_trend = "stable"
        if len(memory_values) >= 3:
            if memory_values[-1] > memory_values[0] * 1.1:
                memory_trend = "increasing"
            elif memory_values[-1] < memory_values[0] * 0.9:
                memory_trend = "decreasing"
        
        # Calculate health trend
        health_scores = []
        for snapshot in recent_snapshots:
            status = snapshot["health_summary"]["overall_status"]
            score = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0, "unknown": 0.25}[status]
            health_scores.append(score)
        
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0
        
        return {
            "memory_trend": memory_trend,
            "avg_health_score": avg_health,
            "snapshots_analyzed": len(recent_snapshots)
        }
    
    def _register_default_alert_rules(self):
        """Register default alert rules for common issues."""
        
        # High memory usage alert
        def high_memory_condition(metrics):
            return metrics.get("system", {}).get("memory_mb", 0) > 1000  # 1GB threshold
        
        high_memory_rule = AlertRule(
            rule_id="high_memory_usage",
            condition_func=high_memory_condition,
            alert_level=AlertLevel.WARNING,
            message_template="High memory usage detected: {system[memory_mb]:.1f}MB",
            component="system",
            cooldown_seconds=300
        )
        
        # High error rate alert
        def high_error_rate_condition(metrics):
            flows = metrics.get("flows", {})
            # This would need to check actual flow error rates
            return False  # Placeholder
        
        high_error_rule = AlertRule(
            rule_id="high_error_rate",
            condition_func=high_error_rate_condition,
            alert_level=AlertLevel.CRITICAL,
            message_template="High error rate detected in flows",
            component="flows",
            cooldown_seconds=180
        )
        
        # Health degradation alert
        def health_degradation_condition(metrics):
            health = metrics.get("health", {})
            return health.get("overall_status") == "unhealthy"
        
        health_degradation_rule = AlertRule(
            rule_id="health_degradation",
            condition_func=health_degradation_condition,
            alert_level=AlertLevel.CRITICAL,
            message_template="System health degraded: {health[overall_status]}",
            component="health",
            cooldown_seconds=120
        )
        
        # Register rules
        self.register_alert_rule(high_memory_rule)
        self.register_alert_rule(high_error_rule)
        self.register_alert_rule(health_degradation_rule)


# Global observability manager instance
_global_observability_manager: Optional[ObservabilityManager] = None

def get_observability_manager() -> ObservabilityManager:
    """Get or create global observability manager instance."""
    global _global_observability_manager
    if _global_observability_manager is None:
        _global_observability_manager = ObservabilityManager()
    return _global_observability_manager

async def shutdown_observability_manager():
    """Shutdown global observability manager."""
    global _global_observability_manager
    if _global_observability_manager:
        await _global_observability_manager.stop_monitoring()
        _global_observability_manager = None