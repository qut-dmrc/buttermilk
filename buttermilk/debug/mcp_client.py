"""MCP client for testing buttermilk flows."""

import time
import requests
from typing import Any
from requests.exceptions import RequestException, Timeout

from buttermilk._core.log import logger
from .models import MCPTestResult, HealthStatus, FlowTestSuite, AgentTestResult, FlowHealthReport


class MCPFlowTester:
    """Test flows using existing MCP endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
    def health_check(self) -> HealthStatus:
        """Check API health and available flows."""
        start_time = time.time()
        
        try:
            # Check basic health
            health_response = self.session.get(
                f"{self.base_url}/monitoring/health",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if health_response.status_code == 200:
                # Get available flows
                flows_response = self.session.get(
                    f"{self.base_url}/api/flows",
                    timeout=self.timeout
                )
                
                available_flows = []
                if flows_response.status_code == 200:
                    flows_data = flows_response.json()
                    # Extract flow names from response
                    if isinstance(flows_data, dict):
                        available_flows = list(flows_data.keys())
                    elif isinstance(flows_data, list):
                        available_flows = flows_data
                
                return HealthStatus(
                    api_reachable=True,
                    status_code=health_response.status_code,
                    response_time=response_time,
                    available_flows=available_flows
                )
            else:
                return HealthStatus(
                    api_reachable=True,
                    status_code=health_response.status_code,
                    response_time=response_time,
                    error_details=f"Health check failed with status {health_response.status_code}"
                )
                
        except Timeout:
            return HealthStatus(
                api_reachable=False,
                response_time=time.time() - start_time,
                error_details="Health check timed out"
            )
        except RequestException as e:
            return HealthStatus(
                api_reachable=False,
                response_time=time.time() - start_time,
                error_details=f"Health check failed: {str(e)}"
            )
    
    def test_mcp_flow_start(self, flow_name: str, query: str) -> MCPTestResult:
        """Test starting a flow via MCP endpoint."""
        start_time = time.time()
        endpoint = f"{self.base_url}/mcp/flows/start"
        
        payload = {
            "flow_name": flow_name,
            "initial_message": query,
            "session_id": f"debug-{int(time.time())}"
        }
        
        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return MCPTestResult(
                    endpoint=endpoint,
                    flow_name=flow_name,
                    query=query,
                    response_time=response_time,
                    status="success",
                    response=data,
                    trace_id=data.get("trace_id")
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
                
        except Timeout:
            return MCPTestResult(
                endpoint=endpoint,
                flow_name=flow_name,
                query=query,
                response_time=time.time() - start_time,
                status="timeout",
                error_details="Request timed out"
            )
        except RequestException as e:
            return MCPTestResult(
                endpoint=endpoint,
                flow_name=flow_name,
                query=query,
                response_time=time.time() - start_time,
                status="error",
                error_details=str(e)
            )
    
    def test_osb_vector_query(self, query: str) -> MCPTestResult:
        """Test OSB vector store query via MCP."""
        start_time = time.time()
        endpoint = f"{self.base_url}/mcp/osb/vector-query"
        
        payload = {
            "query": query,
            "max_results": 5,
            "search_strategy": "semantic"
        }
        
        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return MCPTestResult(
                    endpoint=endpoint,
                    flow_name="osb",
                    query=query,
                    response_time=response_time,
                    status="success",
                    response=data,
                    trace_id=data.get("trace_id")
                )
            else:
                return MCPTestResult(
                    endpoint=endpoint,
                    flow_name="osb",
                    query=query,
                    response_time=response_time,
                    status="error",
                    error_details=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return MCPTestResult(
                endpoint=endpoint,
                flow_name="osb",
                query=query,
                response_time=time.time() - start_time,
                status="error",
                error_details=str(e)
            )
    
    def test_osb_agent(self, agent_role: str, query: str) -> AgentTestResult:
        """Test individual OSB agent via MCP."""
        start_time = time.time()
        endpoint = f"{self.base_url}/mcp/osb/agent-invoke"
        
        payload = {
            "agent_role": agent_role,
            "query": query,
            "include_context": True
        }
        
        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return AgentTestResult(
                    agent_role=agent_role,
                    agent_type="enhanced_rag",
                    test_query=query,
                    response_time=response_time,
                    status="success",
                    response_content=data.get("result", {}).get("content"),
                    vector_search_results=data.get("result", {}).get("search_results_count")
                )
            else:
                return AgentTestResult(
                    agent_role=agent_role,
                    agent_type="enhanced_rag",
                    test_query=query,
                    response_time=response_time,
                    status="error",
                    error_details=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return AgentTestResult(
                agent_role=agent_role,
                agent_type="enhanced_rag",
                test_query=query,
                response_time=time.time() - start_time,
                status="error",
                error_details=str(e)
            )
    
    def test_flow_info(self, flow_name: str) -> MCPTestResult:
        """Test getting flow information."""
        start_time = time.time()
        endpoint = f"{self.base_url}/api/flows/{flow_name}/info"
        
        try:
            response = self.session.get(endpoint, timeout=self.timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return MCPTestResult(
                    endpoint=endpoint,
                    flow_name=flow_name,
                    response_time=response_time,
                    status="success",
                    response=response.json()
                )
            else:
                return MCPTestResult(
                    endpoint=endpoint,
                    flow_name=flow_name,
                    response_time=response_time,
                    status="error",
                    error_details=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return MCPTestResult(
                endpoint=endpoint,
                flow_name=flow_name,
                response_time=time.time() - start_time,
                status="error",
                error_details=str(e)
            )
    
    def test_flow_comprehensive(self, flow_name: str, test_queries: list[str] = None) -> FlowHealthReport:
        """Run comprehensive test suite for a flow."""
        if test_queries is None:
            test_queries = [
                "does hate speech have to be explicit to be prohibited?",
                "what are the key policy considerations for content moderation?",
                "how should platforms handle borderline hate speech cases?"
            ]
        
        logger.info(f"Starting comprehensive test of {flow_name} flow with {len(test_queries)} queries")
        
        # Health check
        health = self.health_check()
        
        # Flow info
        flow_info_result = self.test_flow_info(flow_name)
        flow_info = flow_info_result.response if flow_info_result.status == "success" else None
        
        # Test MCP endpoints
        mcp_results = []
        agent_results = []
        
        if flow_name == "osb":
            # Test OSB-specific endpoints
            for query in test_queries:
                # Test vector query
                vector_result = self.test_osb_vector_query(query)
                mcp_results.append(vector_result)
                
                # Test individual agents
                for agent_role in ["researcher", "policy_analyst", "fact_checker", "explorer"]:
                    agent_result = self.test_osb_agent(agent_role, query)
                    agent_results.append(agent_result)
                
                # Test flow start (only for first query to avoid overload)
                if query == test_queries[0]:
                    flow_start_result = self.test_mcp_flow_start(flow_name, query)
                    mcp_results.append(flow_start_result)
        
        # Generate recommendations
        recommendations = []
        if not health.api_reachable:
            recommendations.append("API is not reachable - check if daemon is running")
        
        failed_mcp_tests = [r for r in mcp_results if r.status != "success"]
        if failed_mcp_tests:
            recommendations.append(f"{len(failed_mcp_tests)} MCP tests failed - check error details")
        
        failed_agent_tests = [r for r in agent_results if r.status != "success"]  
        if failed_agent_tests:
            recommendations.append(f"{len(failed_agent_tests)} agent tests failed - check agent configuration")
        
        if flow_info is None:
            recommendations.append(f"Could not retrieve {flow_name} flow info - check flow configuration")
        
        return FlowHealthReport(
            flow_name=flow_name,
            api_health=health,
            flow_info=flow_info,
            agent_tests=agent_results,
            mcp_test_results=mcp_results,
            recommendations=recommendations
        )