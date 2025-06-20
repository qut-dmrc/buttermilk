"""MCP client for testing buttermilk flows."""

import asyncio
import time
from typing import Any

import httpx
from pydantic import BaseModel

from buttermilk._core.log import logger
from buttermilk.debug.models import (
    AgentTestResult,
    FlowHealthReport,
    FlowTestSuite,
    HealthStatus,
    MCPTestResult,
)


class MCPFlowTester:
    """Test flows using existing MCP endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the MCP client.
        
        Args:
            base_url: Base URL of the buttermilk API
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> HealthStatus:
        """Check API health and available flows.
        
        Returns:
            HealthStatus with API health information
        """
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/api/flows")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                flows = list(data.get("flows", {}).keys())
                return HealthStatus(
                    api_reachable=True,
                    status_code=response.status_code,
                    response_time=response_time,
                    available_flows=flows
                )
            else:
                return HealthStatus(
                    api_reachable=True,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_details=f"Unexpected status code: {response.status_code}"
                )
        except Exception as e:
            return HealthStatus(
                api_reachable=False,
                response_time=time.time() - start_time,
                error_details=str(e)
            )
    
    async def test_flow_query(self, flow_name: str, query: str) -> MCPTestResult:
        """Test a single query using MCP flow endpoint.
        
        Args:
            flow_name: Name of the flow to test
            query: Query string to send
            
        Returns:
            MCPTestResult with test outcome
        """
        endpoint = f"/mcp/flow/{flow_name}"
        start_time = time.time()
        
        try:
            # Prepare the MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": f"flow_{flow_name}",
                    "arguments": {
                        "query": query
                    }
                },
                "id": f"test-{int(time.time() * 1000)}"
            }
            
            logger.info(f"Testing flow '{flow_name}' with query: {query}")
            response = await self.client.post(
                f"{self.base_url}{endpoint}",
                json=mcp_request
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for MCP error response
                if "error" in data:
                    return MCPTestResult(
                        endpoint=endpoint,
                        flow_name=flow_name,
                        query=query,
                        response_time=response_time,
                        status="error",
                        error_details=data["error"].get("message", "Unknown error")
                    )
                
                # Success case
                result = data.get("result", {})
                return MCPTestResult(
                    endpoint=endpoint,
                    flow_name=flow_name,
                    query=query,
                    response_time=response_time,
                    status="success",
                    response=result,
                    trace_id=result.get("content", [{}])[0].get("trace_id")
                )
            else:
                return MCPTestResult(
                    endpoint=endpoint,
                    flow_name=flow_name,
                    query=query,
                    response_time=response_time,
                    status="error",
                    error_details=f"HTTP {response.status_code}: {response.text}"
                )
                
        except asyncio.TimeoutError:
            return MCPTestResult(
                endpoint=endpoint,
                flow_name=flow_name,
                query=query,
                response_time=time.time() - start_time,
                status="timeout",
                error_details="Request timed out after 60 seconds"
            )
        except Exception as e:
            return MCPTestResult(
                endpoint=endpoint,
                flow_name=flow_name,
                query=query,
                response_time=time.time() - start_time,
                status="error",
                error_details=str(e)
            )
    
    async def test_flow_comprehensive(
        self, 
        flow_name: str, 
        test_queries: list[str]
    ) -> FlowTestSuite:
        """Run comprehensive test suite for a flow.
        
        Args:
            flow_name: Name of the flow to test
            test_queries: List of test queries to run
            
        Returns:
            FlowTestSuite with comprehensive results
        """
        logger.info(f"Running comprehensive test suite for flow '{flow_name}'")
        start_time = time.time()
        
        # First, check health
        health = await self.health_check()
        
        # Run all test queries
        test_results = []
        for query in test_queries:
            result = await self.test_flow_query(flow_name, query)
            test_results.append(result)
            
            # Brief delay between tests to avoid overwhelming the server
            await asyncio.sleep(0.5)
        
        # Calculate statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == "success")
        failed_tests = total_tests - passed_tests
        
        response_times = [r.response_time for r in test_results if r.status == "success"]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Generate recommendations
        recommendations = []
        if not health.api_reachable:
            recommendations.append("API is not reachable. Check if the service is running.")
        if failed_tests > 0:
            recommendations.append(f"{failed_tests} tests failed. Review error details.")
        if avg_response_time > 10:
            recommendations.append(f"Average response time ({avg_response_time:.1f}s) is high. Consider performance optimization.")
        
        return FlowTestSuite(
            flow_name=flow_name,
            health_check=health,
            mcp_tests=test_results,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_response_time=avg_response_time,
            recommendations=recommendations
        )
    
    async def test_agent_tool(
        self,
        flow_name: str,
        agent_role: str,
        tool_name: str,
        tool_args: dict[str, Any]
    ) -> AgentTestResult:
        """Test a specific agent tool via MCP.
        
        Args:
            flow_name: Name of the flow
            agent_role: Role of the agent to test
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            
        Returns:
            AgentTestResult with test outcome
        """
        start_time = time.time()
        
        try:
            # Use the agent-specific MCP endpoint
            endpoint = f"/mcp/flow/{flow_name}/agent/{agent_role}"
            
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": tool_args
                },
                "id": f"agent-test-{int(time.time() * 1000)}"
            }
            
            response = await self.client.post(
                f"{self.base_url}{endpoint}",
                json=mcp_request
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    return AgentTestResult(
                        agent_role=agent_role,
                        agent_type=tool_name,
                        test_query=str(tool_args),
                        response_time=response_time,
                        status="error",
                        error_details=data["error"].get("message")
                    )
                
                result = data.get("result", {})
                return AgentTestResult(
                    agent_role=agent_role,
                    agent_type=tool_name,
                    test_query=str(tool_args),
                    response_time=response_time,
                    status="success",
                    response_content=str(result)
                )
            else:
                return AgentTestResult(
                    agent_role=agent_role,
                    agent_type=tool_name,
                    test_query=str(tool_args),
                    response_time=response_time,
                    status="error",
                    error_details=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return AgentTestResult(
                agent_role=agent_role,
                agent_type=tool_name,
                test_query=str(tool_args),
                response_time=time.time() - start_time,
                status="error",
                error_details=str(e)
            )
    
    async def generate_flow_health_report(
        self,
        flow_name: str,
        test_queries: list[str] | None = None
    ) -> FlowHealthReport:
        """Generate a comprehensive health report for a flow.
        
        Args:
            flow_name: Name of the flow to analyze
            test_queries: Optional list of test queries
            
        Returns:
            FlowHealthReport with full analysis
        """
        logger.info(f"Generating health report for flow '{flow_name}'")
        
        # Check API health
        health = await self.health_check()
        
        # Get flow info
        flow_info = None
        try:
            response = await self.client.get(f"{self.base_url}/api/flows/{flow_name}")
            if response.status_code == 200:
                flow_info = response.json()
        except Exception:
            pass
        
        # Run test queries if provided
        mcp_results = []
        if test_queries:
            for query in test_queries:
                result = await self.test_flow_query(flow_name, query)
                mcp_results.append(result)
        
        # Generate recommendations
        recommendations = []
        if not health.api_reachable:
            recommendations.append("API is not reachable. Check service status.")
        elif flow_name not in health.available_flows:
            recommendations.append(f"Flow '{flow_name}' not found in available flows.")
        
        if mcp_results:
            failures = [r for r in mcp_results if r.status != "success"]
            if failures:
                recommendations.append(f"{len(failures)} test(s) failed. Review error details.")
        
        return FlowHealthReport(
            flow_name=flow_name,
            api_health=health,
            flow_info=flow_info,
            mcp_test_results=mcp_results,
            recommendations=recommendations
        )