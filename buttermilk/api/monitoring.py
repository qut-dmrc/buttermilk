"""
Simplified Monitoring API Endpoints.

This module provides basic REST API endpoints for:
- Basic health status checking
- Fatal error detection
- Simple system monitoring

External monitoring tools handle detailed metrics and alerting.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from buttermilk._core import logger
from buttermilk.monitoring import MetricsCollector, SimpleHealthMonitor, get_metrics_collector, get_simple_health_monitor

# Create simplified monitoring router
monitoring_router = APIRouter(prefix="/monitoring", tags=["Basic Monitoring"])


# Response models
class SimpleHealthCheckResponse(BaseModel):
    """Simple health check response model."""
    status: str
    timestamp: str
    overall_status: str
    fatal_error_detected: bool
    fatal_error_message: Optional[str] = None


class BasicMetricsResponse(BaseModel):
    """Basic metrics response model."""
    total_flows: int
    system_uptime_seconds: float
    memory_mb: float
    active_sessions: int
    timestamp: str


# Dependency injection
def get_metrics() -> MetricsCollector:
    """Get metrics collector dependency."""
    return get_metrics_collector()


def get_health_monitor() -> SimpleHealthMonitor:
    """Get simple health monitor dependency.""" 
    return get_simple_health_monitor()


# Basic health check endpoints
@monitoring_router.get("/health", response_model=SimpleHealthCheckResponse)
async def health_check(health_monitor: SimpleHealthMonitor = Depends(get_health_monitor)):
    """
    Get basic system health status.
    
    Returns essential health information focused on fatal errors and basic system status.
    """
    try:
        system_status = health_monitor.get_basic_health_status()

        return SimpleHealthCheckResponse(
            status="ok",
            timestamp=system_status.timestamp.isoformat(),
            overall_status=system_status.overall_status.value,
            fatal_error_detected=health_monitor.check_fatal_errors(),
            fatal_error_message=health_monitor.get_fatal_error_message()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@monitoring_router.get("/fatal-errors")
async def check_fatal_errors(health_monitor: SimpleHealthMonitor = Depends(get_health_monitor)):
    """
    Check for fatal errors that require system exit.
    
    Returns fatal error status and message if any detected.
    """
    try:
        return {
            "fatal_error_detected": health_monitor.check_fatal_errors(),
            "fatal_error_message": health_monitor.get_fatal_error_message(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Fatal error check failed: {e}")
        raise HTTPException(status_code=500, detail="Fatal error check failed")


@monitoring_router.post("/fatal-errors")
async def report_fatal_error(
    error_message: str,
    health_monitor: SimpleHealthMonitor = Depends(get_health_monitor)
):
    """
    Report a fatal error that should cause system exit.
    
    Use this endpoint to report critical errors that require immediate attention.
    """
    try:
        health_monitor.report_fatal_error(error_message)
        return {
            "message": "Fatal error reported",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to report fatal error: {e}")
        raise HTTPException(status_code=500, detail="Failed to report fatal error")


# Basic metrics endpoints
@monitoring_router.get("/metrics/basic", response_model=BasicMetricsResponse)
async def get_basic_metrics(health_monitor: SimpleHealthMonitor = Depends(get_health_monitor)):
    """
    Get basic system metrics.
    
    Returns essential metrics for simple monitoring without complex analytics.
    """
    try:
        metrics_summary = health_monitor.get_basic_metrics_summary()

        return BasicMetricsResponse(
            total_flows=metrics_summary.get("total_flows", 0),
            system_uptime_seconds=metrics_summary.get("system_uptime_seconds", 0),
            memory_mb=metrics_summary.get("memory_mb", 0),
            active_sessions=metrics_summary.get("active_sessions", 0),
            timestamp=metrics_summary["timestamp"]
        )
    except Exception as e:
        logger.error(f"Failed to get basic metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve basic metrics")


# Flow responsiveness checks
@monitoring_router.get("/flows/{flow_name}/responsiveness")
async def check_flow_responsiveness(
    flow_name: str,
    timeout_seconds: int = 300,
    health_monitor: SimpleHealthMonitor = Depends(get_health_monitor)
):
    """
    Check if a flow is responsive.
    
    Returns whether the flow is responding within the specified timeout.
    """
    try:
        is_responsive = health_monitor.check_flow_responsiveness(flow_name, timeout_seconds)
        
        return {
            "flow_name": flow_name,
            "is_responsive": is_responsive,
            "timeout_seconds": timeout_seconds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to check flow responsiveness for {flow_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check flow responsiveness")


@monitoring_router.get("/sessions/{session_id}/ui-timeout")
async def check_interactive_ui_timeout(
    session_id: str,
    timeout_seconds: int = 1800,
    health_monitor: SimpleHealthMonitor = Depends(get_health_monitor)
):
    """
    Check if interactive flow has been without UI for too long.
    
    Returns whether the interactive session is still active or stuck without UI.
    """
    try:
        is_ui_active = health_monitor.check_interactive_flow_ui_timeout(session_id, timeout_seconds)
        
        return {
            "session_id": session_id,
            "ui_active": is_ui_active,
            "timeout_seconds": timeout_seconds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to check UI timeout for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check UI timeout")


# System status endpoint
@monitoring_router.get("/status")
async def get_monitoring_status():
    """
    Get monitoring system status.
    
    Returns information about the simplified monitoring system.
    """
    try:
        return {
            "monitoring_type": "simplified",
            "description": "Basic health monitoring focused on fatal errors and flow responsiveness",
            "external_monitoring_recommended": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring status")
