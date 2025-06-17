"""
Comprehensive Metrics Collection for Flow-Agnostic Monitoring.

This module provides real-time metrics collection for any flow configuration,
tracking performance, usage patterns, and system health across all components.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import asyncio

from buttermilk._core import logger


@dataclass
class FlowMetrics:
    """Metrics for flow execution performance."""
    flow_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0
    execution_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_rate: float = 0.0
    last_execution: Optional[datetime] = None
    throughput_per_minute: float = 0.0
    
    def update_execution(self, execution_time: float, success: bool):
        """Update metrics with new execution data."""
        self.total_executions += 1
        self.execution_times.append(execution_time)
        
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        self.last_execution = datetime.now()
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate derived metrics."""
        if self.execution_times:
            self.avg_execution_time = statistics.mean(self.execution_times)
            sorted_times = sorted(self.execution_times)
            n = len(sorted_times)
            if n >= 20:  # Only calculate percentiles with sufficient data
                self.p95_execution_time = sorted_times[int(n * 0.95)]
                self.p99_execution_time = sorted_times[int(n * 0.99)]
        
        if self.total_executions > 0:
            self.error_rate = self.failed_executions / self.total_executions


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance."""
    agent_name: str
    flow_name: str
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    avg_response_time: float = 0.0
    avg_confidence_score: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=500))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=500))
    error_types: Dict[str, int] = field(default_factory=dict)
    last_invocation: Optional[datetime] = None
    
    def update_invocation(self, response_time: float, success: bool, 
                         confidence: Optional[float] = None, error_type: Optional[str] = None):
        """Update agent metrics with new invocation data."""
        self.total_invocations += 1
        self.response_times.append(response_time)
        self.last_invocation = datetime.now()
        
        if success:
            self.successful_invocations += 1
            if confidence is not None:
                self.confidence_scores.append(confidence)
        else:
            self.failed_invocations += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate derived metrics."""
        if self.response_times:
            self.avg_response_time = statistics.mean(self.response_times)
        
        if self.confidence_scores:
            self.avg_confidence_score = statistics.mean(self.confidence_scores)


@dataclass  
class SessionMetrics:
    """Metrics for session lifecycle and resource usage."""
    session_id: str
    flow_name: str
    created_at: datetime
    last_activity: datetime
    total_queries: int = 0
    total_responses: int = 0
    avg_query_time: float = 0.0
    peak_memory_mb: float = 0.0
    total_data_transferred: int = 0  # bytes
    websocket_reconnections: int = 0
    error_count: int = 0
    is_active: bool = True
    
    def update_activity(self, query_time: Optional[float] = None, 
                       memory_mb: Optional[float] = None,
                       data_bytes: Optional[int] = None,
                       error_occurred: bool = False):
        """Update session activity metrics."""
        self.last_activity = datetime.now()
        
        if query_time is not None:
            self.total_queries += 1
            # Simple moving average for query time
            self.avg_query_time = (self.avg_query_time * (self.total_queries - 1) + query_time) / self.total_queries
        
        if memory_mb is not None:
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        
        if data_bytes is not None:
            self.total_data_transferred += data_bytes
        
        if error_occurred:
            self.error_count += 1
    
    def mark_inactive(self):
        """Mark session as inactive."""
        self.is_active = False


class MetricsCollector:
    """Thread-safe metrics collector for production monitoring."""
    
    def __init__(self, retention_days: int = 7):
        """Initialize metrics collector with retention policy."""
        self.retention_days = retention_days
        self._lock = threading.RLock()
        
        # Flow-level metrics
        self.flow_metrics: Dict[str, FlowMetrics] = {}
        
        # Agent-level metrics
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # Session-level metrics
        self.session_metrics: Dict[str, SessionMetrics] = {}
        
        # System-level metrics
        self.system_metrics = {
            "start_time": datetime.now(),
            "total_memory_mb": 0.0,
            "peak_memory_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "active_sessions": 0,
            "total_sessions_created": 0,
            "websocket_connections": 0
        }
        
        # Time-series data for trending
        self.time_series_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours at minute resolution
        
        # Cleanup thread
        self._cleanup_thread = None
        self._should_cleanup = True
        self._start_cleanup_thread()
    
    def record_flow_execution(self, flow_name: str, execution_time: float, success: bool):
        """Record flow execution metrics."""
        with self._lock:
            if flow_name not in self.flow_metrics:
                self.flow_metrics[flow_name] = FlowMetrics(flow_name=flow_name)
            
            self.flow_metrics[flow_name].update_execution(execution_time, success)
            
            # Update time series
            timestamp = datetime.now()
            self.time_series_data[f"flow_execution_{flow_name}"].append({
                "timestamp": timestamp,
                "execution_time": execution_time,
                "success": success
            })
            
            logger.debug(f"Recorded flow execution: {flow_name}, time: {execution_time:.2f}s, success: {success}")
    
    def record_agent_invocation(self, agent_name: str, flow_name: str, 
                               response_time: float, success: bool,
                               confidence: Optional[float] = None,
                               error_type: Optional[str] = None):
        """Record agent invocation metrics."""
        with self._lock:
            agent_key = f"{flow_name}.{agent_name}"
            if agent_key not in self.agent_metrics:
                self.agent_metrics[agent_key] = AgentMetrics(
                    agent_name=agent_name, 
                    flow_name=flow_name
                )
            
            self.agent_metrics[agent_key].update_invocation(
                response_time, success, confidence, error_type
            )
            
            # Update time series
            timestamp = datetime.now()
            self.time_series_data[f"agent_invocation_{agent_key}"].append({
                "timestamp": timestamp,
                "response_time": response_time,
                "success": success,
                "confidence": confidence
            })
            
            logger.debug(f"Recorded agent invocation: {agent_key}, time: {response_time:.2f}s")
    
    def start_session_tracking(self, session_id: str, flow_name: str):
        """Start tracking a new session."""
        with self._lock:
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
    
    def update_session_activity(self, session_id: str, **kwargs):
        """Update session activity metrics."""
        with self._lock:
            if session_id in self.session_metrics:
                self.session_metrics[session_id].update_activity(**kwargs)
    
    def end_session_tracking(self, session_id: str):
        """End session tracking and mark as inactive."""
        with self._lock:
            if session_id in self.session_metrics:
                self.session_metrics[session_id].mark_inactive()
                self._update_active_session_count()
                
                logger.info(f"Ended session tracking: {session_id}")
    
    def update_system_metrics(self, memory_mb: float, cpu_percent: float, 
                             websocket_connections: int):
        """Update system-level metrics."""
        with self._lock:
            self.system_metrics["total_memory_mb"] = memory_mb
            self.system_metrics["peak_memory_mb"] = max(
                self.system_metrics["peak_memory_mb"], memory_mb
            )
            self.system_metrics["cpu_usage_percent"] = cpu_percent
            self.system_metrics["websocket_connections"] = websocket_connections
            
            # Update time series
            timestamp = datetime.now()
            self.time_series_data["system_metrics"].append({
                "timestamp": timestamp,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "websocket_connections": websocket_connections
            })
    
    def get_flow_metrics(self, flow_name: Optional[str] = None) -> Dict[str, FlowMetrics]:
        """Get flow metrics for specific flow or all flows."""
        with self._lock:
            if flow_name:
                return {flow_name: self.flow_metrics.get(flow_name, FlowMetrics(flow_name=flow_name))}
            return dict(self.flow_metrics)
    
    def get_agent_metrics(self, flow_name: Optional[str] = None) -> Dict[str, AgentMetrics]:
        """Get agent metrics for specific flow or all agents."""
        with self._lock:
            if flow_name:
                return {k: v for k, v in self.agent_metrics.items() if v.flow_name == flow_name}
            return dict(self.agent_metrics)
    
    def get_session_metrics(self, active_only: bool = False) -> Dict[str, SessionMetrics]:
        """Get session metrics, optionally filtered to active sessions only."""
        with self._lock:
            if active_only:
                return {k: v for k, v in self.session_metrics.items() if v.is_active}
            return dict(self.session_metrics)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self._lock:
            return dict(self.system_metrics)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics summary report."""
        with self._lock:
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
                    "avg_confidence": metrics.avg_confidence_score,
                    "success_rate": (metrics.successful_invocations / metrics.total_invocations
                                   if metrics.total_invocations > 0 else 0.0)
                }
            
            # Session summary
            active_sessions = len([s for s in self.session_metrics.values() if s.is_active])
            
            return {
                "system": {
                    "uptime_seconds": uptime.total_seconds(),
                    "total_memory_mb": self.system_metrics["total_memory_mb"],
                    "peak_memory_mb": self.system_metrics["peak_memory_mb"],
                    "cpu_usage_percent": self.system_metrics["cpu_usage_percent"],
                    "websocket_connections": self.system_metrics["websocket_connections"]
                },
                "flows": flow_summary,
                "agents": agent_summary,
                "sessions": {
                    "active_sessions": active_sessions,
                    "total_sessions_created": self.system_metrics["total_sessions_created"]
                },
                "generated_at": datetime.now().isoformat()
            }
    
    def get_time_series_data(self, metric_name: str, 
                            duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get time series data for specific metric."""
        with self._lock:
            if metric_name not in self.time_series_data:
                return []
            
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            return [
                data for data in self.time_series_data[metric_name]
                if data["timestamp"] >= cutoff_time
            ]
    
    def _update_active_session_count(self):
        """Update active session count in system metrics."""
        active_count = len([s for s in self.session_metrics.values() if s.is_active])
        self.system_metrics["active_sessions"] = active_count
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while self._should_cleanup:
                try:
                    self._cleanup_old_data()
                    time.sleep(3600)  # Cleanup every hour
                except Exception as e:
                    logger.error(f"Error in metrics cleanup: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_old_data(self):
        """Clean up old metrics data based on retention policy."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        with self._lock:
            # Clean up inactive sessions older than retention period
            sessions_to_remove = [
                session_id for session_id, metrics in self.session_metrics.items()
                if not metrics.is_active and metrics.last_activity < cutoff_time
            ]
            
            for session_id in sessions_to_remove:
                del self.session_metrics[session_id]
                logger.debug(f"Cleaned up old session metrics: {session_id}")
            
            # Clean up time series data older than retention period
            for metric_name, data_deque in self.time_series_data.items():
                # Remove old data points
                while data_deque and data_deque[0]["timestamp"] < cutoff_time:
                    data_deque.popleft()
    
    def shutdown(self):
        """Shutdown metrics collector and cleanup resources."""
        self._should_cleanup = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        
        logger.info("Metrics collector shutdown completed")


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
        _global_metrics_collector.shutdown()
        _global_metrics_collector = None