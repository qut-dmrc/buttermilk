"""
Production Monitoring API Endpoints.

This module provides REST API endpoints for accessing monitoring data,
health status, alerts, and performance metrics for production operations.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from buttermilk._core import logger
from buttermilk.monitoring import (
    get_metrics_collector, 
    get_observability_manager,
    MetricsCollector,
    ObservabilityManager,
    HealthStatus,
    AlertLevel
)
from buttermilk.monitoring.health_monitor import HealthMonitor


# Create monitoring router
monitoring_router = APIRouter(prefix="/monitoring", tags=["Production Monitoring"])


# Response models
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    components: Dict[str, Any]
    overall_status: str


class MetricsSummaryResponse(BaseModel):
    """Metrics summary response model."""
    system: Dict[str, Any]
    flows: Dict[str, Any]
    agents: Dict[str, Any]
    sessions: Dict[str, Any]
    generated_at: str


class AlertResponse(BaseModel):
    """Alert response model."""
    alert_id: str
    level: str
    component: str
    message: str
    timestamp: str
    resolved: bool
    acknowledged: bool


class DashboardDataResponse(BaseModel):
    """Dashboard data response model."""
    timestamp: str
    system: Dict[str, Any]
    flows: Dict[str, Any]
    health: Dict[str, Any]
    alerts: Dict[str, Any]
    performance_trends: Dict[str, Any]


# Dependency injection
def get_metrics() -> MetricsCollector:
    """Get metrics collector dependency."""
    return get_metrics_collector()


def get_observability() -> ObservabilityManager:
    """Get observability manager dependency."""
    return get_observability_manager()


# Health check endpoints
@monitoring_router.get("/health", response_model=HealthCheckResponse)
async def health_check(observability: ObservabilityManager = Depends(get_observability)):
    """
    Get overall system health status.
    
    Returns comprehensive health information for all monitored components
    including system resources, database connectivity, vector stores, and flows.
    """
    try:
        health_summary = observability.health_monitor.get_health_summary()
        
        return HealthCheckResponse(
            status="ok",
            timestamp=datetime.now().isoformat(),
            components=health_summary["components"],
            overall_status=health_summary["overall_status"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@monitoring_router.get("/health/{component_name}")
async def component_health_check(
    component_name: str,
    observability: ObservabilityManager = Depends(get_observability)
):
    """
    Get health status for specific component.
    
    Returns detailed health information for a specific system component.
    """
    try:
        component_health = observability.health_monitor.get_component_health(component_name)
        
        if component_health is None:
            raise HTTPException(status_code=404, detail=f"Component '{component_name}' not found")
        
        return {
            "component_name": component_health.component_name,
            "status": component_health.status.value,
            "last_check": component_health.last_check.isoformat() if component_health.last_check else None,
            "response_time_ms": component_health.response_time_ms,
            "error_message": component_health.error_message,
            "uptime_percentage": component_health.uptime_percentage,
            "consecutive_failures": component_health.consecutive_failures,
            "metadata": component_health.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Component health check failed for {component_name}: {e}")
        raise HTTPException(status_code=500, detail="Component health check failed")


# Metrics endpoints
@monitoring_router.get("/metrics", response_model=MetricsSummaryResponse)
async def get_metrics_summary(metrics: MetricsCollector = Depends(get_metrics)):
    """
    Get comprehensive metrics summary.
    
    Returns aggregated metrics for system performance, flow execution,
    agent performance, and session statistics.
    """
    try:
        summary = metrics.get_summary_report()
        
        return MetricsSummaryResponse(
            system=summary["system"],
            flows=summary["flows"],
            agents=summary["agents"],
            sessions=summary["sessions"],
            generated_at=summary["generated_at"]
        )
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@monitoring_router.get("/metrics/flows")
async def get_flow_metrics(
    flow_name: Optional[str] = Query(None, description="Specific flow name to filter"),
    metrics: MetricsCollector = Depends(get_metrics)
):
    """
    Get flow execution metrics.
    
    Returns detailed metrics for flow executions, optionally filtered by flow name.
    """
    try:
        flow_metrics = metrics.get_flow_metrics(flow_name)
        
        result = {}
        for name, flow_data in flow_metrics.items():
            result[name] = {
                "flow_name": flow_data.flow_name,
                "total_executions": flow_data.total_executions,
                "successful_executions": flow_data.successful_executions,
                "failed_executions": flow_data.failed_executions,
                "avg_execution_time": flow_data.avg_execution_time,
                "p95_execution_time": flow_data.p95_execution_time,
                "p99_execution_time": flow_data.p99_execution_time,
                "error_rate": flow_data.error_rate,
                "last_execution": flow_data.last_execution.isoformat() if flow_data.last_execution else None,
                "throughput_per_minute": flow_data.throughput_per_minute
            }
        
        return result
    except Exception as e:
        logger.error(f"Failed to get flow metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve flow metrics")


@monitoring_router.get("/metrics/agents")
async def get_agent_metrics(
    flow_name: Optional[str] = Query(None, description="Filter agents by flow name"),
    metrics: MetricsCollector = Depends(get_metrics)
):
    """
    Get agent performance metrics.
    
    Returns detailed metrics for agent invocations, optionally filtered by flow.
    """
    try:
        agent_metrics = metrics.get_agent_metrics(flow_name)
        
        result = {}
        for agent_key, agent_data in agent_metrics.items():
            result[agent_key] = {
                "agent_name": agent_data.agent_name,
                "flow_name": agent_data.flow_name,
                "total_invocations": agent_data.total_invocations,
                "successful_invocations": agent_data.successful_invocations,
                "failed_invocations": agent_data.failed_invocations,
                "avg_response_time": agent_data.avg_response_time,
                "avg_confidence_score": agent_data.avg_confidence_score,
                "error_types": agent_data.error_types,
                "last_invocation": agent_data.last_invocation.isoformat() if agent_data.last_invocation else None
            }
        
        return result
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve agent metrics")


@monitoring_router.get("/metrics/sessions")
async def get_session_metrics(
    active_only: bool = Query(False, description="Return only active sessions"),
    metrics: MetricsCollector = Depends(get_metrics)
):
    """
    Get session lifecycle metrics.
    
    Returns metrics for session management, optionally filtered to active sessions only.
    """
    try:
        session_metrics = metrics.get_session_metrics(active_only)
        
        result = {}
        for session_id, session_data in session_metrics.items():
            result[session_id] = {
                "session_id": session_data.session_id,
                "flow_name": session_data.flow_name,
                "created_at": session_data.created_at.isoformat(),
                "last_activity": session_data.last_activity.isoformat(),
                "total_queries": session_data.total_queries,
                "total_responses": session_data.total_responses,
                "avg_query_time": session_data.avg_query_time,
                "peak_memory_mb": session_data.peak_memory_mb,
                "total_data_transferred": session_data.total_data_transferred,
                "websocket_reconnections": session_data.websocket_reconnections,
                "error_count": session_data.error_count,
                "is_active": session_data.is_active
            }
        
        return result
    except Exception as e:
        logger.error(f"Failed to get session metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session metrics")


@monitoring_router.get("/metrics/prometheus")
async def get_prometheus_metrics(observability: ObservabilityManager = Depends(get_observability)):
    """
    Get metrics in Prometheus format.
    
    Returns all metrics formatted for Prometheus scraping.
    """
    try:
        prometheus_data = observability.export_metrics_for_prometheus()
        
        # Return as plain text with proper content type
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=prometheus_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to export Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export Prometheus metrics")


# Alert endpoints
@monitoring_router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    level: Optional[str] = Query(None, description="Filter by alert level"),
    active_only: bool = Query(True, description="Return only active alerts"),
    observability: ObservabilityManager = Depends(get_observability)
):
    """
    Get system alerts.
    
    Returns current system alerts, optionally filtered by level and active status.
    """
    try:
        alert_level = None
        if level:
            try:
                alert_level = AlertLevel(level.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid alert level: {level}")
        
        if active_only:
            alerts = observability.get_active_alerts(alert_level)
        else:
            # For simplicity, we'll return active alerts for now
            # In production, you'd have a method to get all alerts including resolved ones
            alerts = observability.get_active_alerts(alert_level)
        
        return [
            AlertResponse(
                alert_id=alert.alert_id,
                level=alert.level.value,
                component=alert.component,
                message=alert.message,
                timestamp=alert.timestamp.isoformat(),
                resolved=alert.resolved,
                acknowledged=alert.acknowledged
            )
            for alert in alerts
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@monitoring_router.get("/alerts/summary")
async def get_alert_summary(observability: ObservabilityManager = Depends(get_observability)):
    """
    Get alert summary statistics.
    
    Returns aggregated alert statistics including counts by level and status.
    """
    try:
        alert_summary = observability.get_alert_summary()
        return alert_summary
    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert summary")


@monitoring_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query(..., description="User acknowledging the alert"),
    observability: ObservabilityManager = Depends(get_observability)
):
    """
    Acknowledge an alert.
    
    Mark an alert as acknowledged by a specific user.
    """
    try:
        success = observability.acknowledge_alert(alert_id, acknowledged_by)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
        return {"message": f"Alert {alert_id} acknowledged by {acknowledged_by}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@monitoring_router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    observability: ObservabilityManager = Depends(get_observability)
):
    """
    Resolve an alert.
    
    Mark an alert as resolved.
    """
    try:
        success = observability.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
        return {"message": f"Alert {alert_id} resolved"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


# Dashboard endpoints
@monitoring_router.get("/dashboard", response_model=DashboardDataResponse)
async def get_dashboard_data(observability: ObservabilityManager = Depends(get_observability)):
    """
    Get comprehensive dashboard data.
    
    Returns aggregated data for monitoring dashboards including system status,
    performance metrics, health information, and alerts.
    """
    try:
        dashboard_data = observability.get_dashboard_data()
        
        return DashboardDataResponse(
            timestamp=dashboard_data["timestamp"],
            system=dashboard_data["system"],
            flows=dashboard_data["flows"],
            health=dashboard_data["health"],
            alerts=dashboard_data["alerts"],
            performance_trends=dashboard_data["performance_trends"]
        )
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@monitoring_router.get("/performance/trends")
async def get_performance_trends(
    duration_hours: int = Query(24, description="Duration in hours for trend analysis"),
    observability: ObservabilityManager = Depends(get_observability)
):
    """
    Get performance trend data.
    
    Returns historical performance data for trend analysis and visualization.
    """
    try:
        # This would return time-series data for performance trends
        # For now, return basic trend information
        dashboard_data = observability.get_dashboard_data()
        
        return {
            "duration_hours": duration_hours,
            "trends": dashboard_data["performance_trends"],
            "current_status": dashboard_data["health"]["overall_status"],
            "data_points": "Limited sample data for demo"  # In production, return actual time series
        }
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance trends")


# System control endpoints
@monitoring_router.post("/system/start-monitoring")
async def start_monitoring(observability: ObservabilityManager = Depends(get_observability)):
    """
    Start system monitoring.
    
    Begins continuous monitoring of system health and performance.
    """
    try:
        await observability.start_monitoring()
        return {"message": "Monitoring started successfully"}
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")


@monitoring_router.post("/system/stop-monitoring")
async def stop_monitoring(observability: ObservabilityManager = Depends(get_observability)):
    """
    Stop system monitoring.
    
    Stops continuous monitoring of system health and performance.
    """
    try:
        await observability.stop_monitoring()
        return {"message": "Monitoring stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")


@monitoring_router.get("/system/status")
async def get_monitoring_status(observability: ObservabilityManager = Depends(get_observability)):
    """
    Get monitoring system status.
    
    Returns information about the monitoring system itself.
    """
    try:
        return {
            "monitoring_active": observability.is_monitoring,
            "health_monitoring_active": observability.health_monitor.is_monitoring,
            "registered_alert_rules": len(observability.alert_rules),
            "registered_alert_handlers": len(observability.alert_handlers),
            "active_components": len(observability.health_monitor.component_health),
            "snapshot_interval": observability.snapshot_interval
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring status")