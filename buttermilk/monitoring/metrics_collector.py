"""
Simplified Metrics Collection for Basic Monitoring.

This module provides essential metrics collection focused on:
- Basic flow execution tracking
- Simple agent invocation recording
- Essential session management
- Minimal system metrics

Complex analytics, time-series data, and P95/P99 calculations are handled externally.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from buttermilk._core import logger


@dataclass
class FlowMetrics:
    """Basic metrics for flow execution performance."""
    flow_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    error_rate: float = 0.0
    last_execution: Optional[datetime] = None

    def update_execution(self, execution_time: float, success: bool):
        """Update metrics with new execution data."""
        self.total_executions += 1
        
        # Simple moving average for execution time
        self.avg_execution_time = (
            (self.avg_execution_time * (self.total_executions - 1) + execution_time) 
            / self.total_executions
        )

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.last_execution = datetime.now()
        
        # Update error rate
        if self.total_executions > 0:
            self.error_rate = self.failed_executions / self.total_executions


@dataclass 
class AgentMetrics:
    """Basic metrics for individual agent performance."""
    agent_name: str
    flow_name: str
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    avg_response_time: float = 0.0
    last_invocation: Optional[datetime] = None

    def update_invocation(self, response_time: float, success: bool):
        """Update agent metrics with new invocation data."""
        self.total_invocations += 1
        
        # Simple moving average for response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_invocations - 1) + response_time) 
            / self.total_invocations
        )
        
        self.last_invocation = datetime.now()

        if success:
            self.successful_invocations += 1
        else:
            self.failed_invocations += 1


@dataclass
class SessionMetrics:
    """Basic metrics for session lifecycle."""
    session_id: str
    flow_name: str
    created_at: datetime
    last_activity: datetime
    total_queries: int = 0
    error_count: int = 0
    is_active: bool = True

    def update_activity(self, error_occurred: bool = False):
        """Update session activity metrics."""
        self.last_activity = datetime.now()
        self.total_queries += 1

        if error_occurred:
            self.error_count += 1

    def mark_inactive(self):
        """Mark session as inactive."""
        self.is_active = False


class MetricsCollector:
    """Simplified metrics collector for basic monitoring."""

    def __init__(self):
        """Initialize simplified metrics collector."""
        # Flow-level metrics
        self.flow_metrics: Dict[str, FlowMetrics] = {}

        # Agent-level metrics 
        self.agent_metrics: Dict[str, AgentMetrics] = {}

        # Session-level metrics
        self.session_metrics: Dict[str, SessionMetrics] = {}

        # Basic system-level metrics
        self.system_metrics = {
            "start_time": datetime.now(),
            "total_memory_mb": 0.0,
            "active_sessions": 0,
            "total_sessions_created": 0
        }

    def record_flow_execution(self, flow_name: str, execution_time: float, success: bool):
        """Record flow execution metrics."""
        if flow_name not in self.flow_metrics:
            self.flow_metrics[flow_name] = FlowMetrics(flow_name=flow_name)

        self.flow_metrics[flow_name].update_execution(execution_time, success)
        logger.debug(f"Recorded flow execution: {flow_name}, time: {execution_time:.2f}s, success: {success}")

    def record_agent_invocation(self, agent_name: str, flow_name: str,
                               response_time: float, success: bool):
        """Record agent invocation metrics."""
        agent_key = f"{flow_name}.{agent_name}"
        if agent_key not in self.agent_metrics:
            self.agent_metrics[agent_key] = AgentMetrics(
                agent_name=agent_name,
                flow_name=flow_name
            )

        self.agent_metrics[agent_key].update_invocation(response_time, success)
        logger.debug(f"Recorded agent invocation: {agent_key}, time: {response_time:.2f}s")

    def start_session_tracking(self, session_id: str, flow_name: str):
        """Start tracking a new session."""
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = SessionMetrics(
                session_id=session_id,
                flow_name=flow_name,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )

            self.system_metrics["total_sessions_created"] += 1
            self._update_active_session_count()

            logger.info(f"Started session tracking: {session_id} for flow: {flow_name}")

    def update_session_activity(self, session_id: str, error_occurred: bool = False):
        """Update session activity metrics."""
        if session_id in self.session_metrics:
            self.session_metrics[session_id].update_activity(error_occurred=error_occurred)

    def end_session_tracking(self, session_id: str):
        """End session tracking and mark as inactive."""
        if session_id in self.session_metrics:
            self.session_metrics[session_id].mark_inactive()
            self._update_active_session_count()

            logger.info(f"Ended session tracking: {session_id}")

    def update_system_metrics(self, memory_mb: float, cpu_percent: Optional[float] = None, websocket_connections: Optional[int] = None):
        """Update basic system-level metrics. Extra parameters ignored for compatibility."""
        self.system_metrics["total_memory_mb"] = memory_mb
        # Ignore cpu_percent and websocket_connections for simplified version

    def get_flow_metrics(self, flow_name: Optional[str] = None) -> Dict[str, FlowMetrics]:
        """Get flow metrics for specific flow or all flows."""
        if flow_name:
            return {flow_name: self.flow_metrics.get(flow_name, FlowMetrics(flow_name=flow_name))}
        return dict(self.flow_metrics)

    def get_agent_metrics(self, flow_name: Optional[str] = None) -> Dict[str, AgentMetrics]:
        """Get agent metrics for specific flow or all agents."""
        if flow_name:
            return {k: v for k, v in self.agent_metrics.items() if v.flow_name == flow_name}
        return dict(self.agent_metrics)

    def get_session_metrics(self, active_only: bool = False) -> Dict[str, SessionMetrics]:
        """Get session metrics, optionally filtered to active sessions only."""
        if active_only:
            return {k: v for k, v in self.session_metrics.items() if v.is_active}
        return dict(self.session_metrics)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current basic system metrics."""
        return dict(self.system_metrics)

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate basic metrics summary report."""
        uptime = datetime.now() - self.system_metrics["start_time"]

        # Flow summary
        flow_summary = {}
        for flow_name, metrics in self.flow_metrics.items():
            flow_summary[flow_name] = {
                "total_executions": metrics.total_executions,
                "success_rate": (metrics.successful_executions / metrics.total_executions
                               if metrics.total_executions > 0 else 0.0),
                "avg_execution_time": metrics.avg_execution_time,
                "error_rate": metrics.error_rate
            }

        # Agent summary
        agent_summary = {}
        for agent_key, metrics in self.agent_metrics.items():
            agent_summary[agent_key] = {
                "total_invocations": metrics.total_invocations,
                "avg_response_time": metrics.avg_response_time,
                "success_rate": (metrics.successful_invocations / metrics.total_invocations
                               if metrics.total_invocations > 0 else 0.0)
            }

        # Session summary
        active_sessions = len([s for s in self.session_metrics.values() if s.is_active])

        return {
            "system": {
                "uptime_seconds": uptime.total_seconds(),
                "total_memory_mb": self.system_metrics["total_memory_mb"],
                "active_sessions": active_sessions
            },
            "flows": flow_summary,
            "agents": agent_summary,
            "sessions": {
                "active_sessions": active_sessions,
                "total_sessions_created": self.system_metrics["total_sessions_created"]
            },
            "generated_at": datetime.now().isoformat()
        }

    def _update_active_session_count(self):
        """Update active session count in system metrics."""
        active_count = len([s for s in self.session_metrics.values() if s.is_active])
        self.system_metrics["active_sessions"] = active_count


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector

def shutdown_metrics_collector():
    """Shutdown global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector:
        # No complex shutdown needed for simplified version
        logger.info("Metrics collector shutdown completed")
        _global_metrics_collector = None
