"""Pydantic models for debug infrastructure."""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class MCPTestResult(BaseModel):
    """Result of testing an MCP endpoint."""
    
    endpoint: str
    flow_name: str | None = None
    query: str | None = None
    response_time: float
    status: Literal["success", "error", "timeout"]
    response: dict | None = None
    error_details: str | None = None
    trace_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthStatus(BaseModel):
    """Health status of the API."""
    
    api_reachable: bool
    status_code: int | None = None
    response_time: float
    available_flows: list[str] = Field(default_factory=list)
    error_details: str | None = None


class FlowTestSuite(BaseModel):
    """Results of comprehensive flow testing."""
    
    flow_name: str
    health_check: HealthStatus
    mcp_tests: list[MCPTestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_response_time: float
    recommendations: list[str] = Field(default_factory=list)


class AgentTestResult(BaseModel):
    """Result of testing individual agent via MCP."""
    
    agent_role: str
    agent_type: str
    test_query: str
    response_time: float
    status: Literal["success", "error", "timeout"]
    response_content: str | None = None
    error_details: str | None = None
    vector_search_results: int | None = None


class FlowHealthReport(BaseModel):
    """Comprehensive health report for a flow."""
    
    flow_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    api_health: HealthStatus
    flow_info: dict | None = None
    agent_tests: list[AgentTestResult] = Field(default_factory=list)
    mcp_test_results: list[MCPTestResult] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    
    @property
    def overall_status(self) -> Literal["healthy", "degraded", "unhealthy"]:
        """Calculate overall health status."""
        if not self.api_health.api_reachable:
            return "unhealthy"
        
        total_tests = len(self.mcp_test_results)
        if total_tests == 0:
            return "degraded"
            
        failed_tests = sum(1 for test in self.mcp_test_results if test.status != "success")
        failure_rate = failed_tests / total_tests
        
        if failure_rate == 0:
            return "healthy"
        elif failure_rate < 0.5:
            return "degraded"
        else:
            return "unhealthy"


class StartupTestResult(BaseModel):
    """Result of testing daemon startup."""
    
    startup_time: float
    startup_success: bool
    validation_errors: list[str] = Field(default_factory=list)
    api_reachable: bool = False
    flows_loaded: list[str] = Field(default_factory=list)
    error_details: str | None = None