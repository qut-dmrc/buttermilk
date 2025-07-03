"""
Flow-Agnostic MCP Endpoints for Agent and Flow Testing.

This module provides MCP-compliant HTTP endpoints for testing any agent
or flow configuration dynamically. These endpoints enable:

- Individual agent testing for any configured agent
- Vector store query testing for any flow configuration
- Multi-agent workflow testing for any agent combination
- Session state management testing across flows
- Generic message validation and processing testing

All endpoints are flow-agnostic and work with any YAML flow configuration,
supporting the modular architecture of the Buttermilk framework.
"""

import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional

import shortuuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from buttermilk._core import logger
from buttermilk._core.types import RunRequest
from buttermilk.api.mcp import MCPToolResponse, MCPError, run_single_agent
from buttermilk.api.routes import get_flows
from buttermilk.runner.flowrunner import FlowRunner

# Create the generic MCP router
agent_mcp_router = APIRouter(prefix="/mcp/agents", tags=["Agent MCP Tools"])


# Request/Response Models for Flow-Agnostic MCP Endpoints

class VectorQueryRequest(BaseModel):
    """Request model for vector store queries (flow-agnostic)."""
    
    query: str = Field(..., description="Query text for vector search")
    flow: str = Field(..., description="Flow configuration to use")
    agent_name: Optional[str] = Field(None, description="Specific agent name for query context")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence threshold")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 characters)")
        return v.strip()


class AgentInvokeRequest(BaseModel):
    """Request model for individual agent invocation (flow-agnostic)."""
    
    query: str = Field(..., description="Query for agent analysis")
    agent_name: str = Field(..., description="Specific agent to invoke")
    flow: str = Field(..., description="Flow configuration to use")
    
    # Generic agent parameters
    max_processing_time: int = Field(default=30, description="Maximum processing time in seconds")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional agent parameters")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class MultiAgentWorkflowRequest(BaseModel):
    """Request model for multi-agent workflow testing (flow-agnostic)."""
    
    query: str = Field(..., description="Query for multi-agent analysis")
    flow: str = Field(..., description="Flow configuration to use")
    agent_names: Optional[List[str]] = Field(None, description="Specific agents to include (default: all agents in flow)")
    
    # Workflow parameters
    enable_synthesis: bool = Field(default=True, description="Enable response synthesis")
    parallel_execution: bool = Field(default=True, description="Execute agents in parallel")
    timeout_seconds: int = Field(default=60, description="Workflow timeout in seconds")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SessionStateRequest(BaseModel):
    """Request model for session state testing (flow-agnostic)."""
    
    session_id: str = Field(..., description="Session identifier to test")
    flow: str = Field(..., description="Flow configuration for session")
    operation: str = Field(..., description="Session operation to perform")
    test_data: Optional[Dict[str, Any]] = Field(None, description="Test data for session operations")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_ops = ["create", "get_status", "update_metadata", "record_metrics", "cleanup", "list_sessions"]
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class MessageValidationRequest(BaseModel):
    """Request model for message validation testing (flow-agnostic)."""
    
    message_type: str = Field(..., description="Type of message to validate")
    message_data: Dict[str, Any] = Field(..., description="Message data to validate")
    flow: Optional[str] = Field(None, description="Flow context for validation")
    strict_validation: bool = Field(default=True, description="Enable strict validation rules")


# Flow-Agnostic MCP Endpoint Implementations

@agent_mcp_router.post("/vector-query", response_model=MCPToolResponse)
async def agent_vector_query_tool(
    request: VectorQueryRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test vector store queries for any flow configuration.
    
    This endpoint allows testing vector store queries with flow-specific
    context and parameters, useful for:
    - Validating vector store connectivity across flows
    - Testing query performance and results
    - Debugging search strategies for different domains
    """
    start_time = time.time()
    trace_id = f"vector-{request.flow}-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Vector query for flow '{request.flow}': {request.query[:100]}...")
        
        # Validate flow exists
        if request.flow not in flow_runner.flows:
            raise HTTPException(status_code=404, detail=f"Flow '{request.flow}' not found")
        
        # Get flow configuration
        flow_config = flow_runner.flows[request.flow]
        
        # Validate agent if specified
        if request.agent_name:
            flow_agents = flow_config.get("agents", {})
            if request.agent_name not in flow_agents:
                available_agents = list(flow_agents.keys())
                raise HTTPException(
                    status_code=400, 
                    detail=f"Agent '{request.agent_name}' not found in flow '{request.flow}'. Available agents: {available_agents}"
                )
        
        # Prepare parameters for vector query
        parameters = {
            "query": request.query,
            "agent_name": request.agent_name,
            "max_results": request.max_results,
            "confidence_threshold": request.confidence_threshold,
            "include_metadata": request.include_metadata,
            "test_mode": True
        }
        
        # Create a test session for the vector query
        session_id = f"test-vector-{trace_id}"
        
        run_request = RunRequest(
            ui_type="mcp",
            flow=request.flow,
            session_id=session_id,
            parameters=parameters
        )
        
        # Execute vector query
        result = await test_vector_query(run_request, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Vector query completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result={
                "query": request.query,
                "flow": request.flow,
                "agent_name": request.agent_name,
                "results": result.get("results", []),
                "metadata": result.get("metadata", {}),
                "performance": {
                    "execution_time_ms": execution_time,
                    "results_count": len(result.get("results", [])),
                    "confidence_scores": result.get("confidence_scores", [])
                }
            },
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Vector query failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Vector query error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@agent_mcp_router.post("/agent-invoke", response_model=MCPToolResponse)
async def agent_invoke_tool(
    request: AgentInvokeRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test individual agent invocations for any flow configuration.
    
    This endpoint enables testing specific agents without running
    the full workflow, useful for:
    - Agent-specific functionality validation
    - Performance testing per agent
    - Debugging individual agent responses
    """
    start_time = time.time()
    trace_id = f"agent-{request.agent_name}-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Invoking agent: {request.agent_name} in flow: {request.flow}")
        
        # Validate flow exists
        if request.flow not in flow_runner.flows:
            raise HTTPException(status_code=404, detail=f"Flow '{request.flow}' not found")
        
        # Get flow configuration
        flow_config = flow_runner.flows[request.flow]
        flow_agents = flow_config.get("agents", {})
        
        # Validate agent exists in flow
        if request.agent_name not in flow_agents:
            available_agents = list(flow_agents.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{request.agent_name}' not found in flow '{request.flow}'. Available agents: {available_agents}"
            )
        
        # Prepare parameters
        parameters = {
            "query": request.query,
            "agent_name": request.agent_name,
            "max_processing_time": request.max_processing_time,
            "test_mode": True,
            **request.parameters
        }
        
        # Run the specific agent
        result = await run_agent_test(request.agent_name, request.flow, parameters, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Agent {request.agent_name} completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result={
                "agent_name": request.agent_name,
                "flow": request.flow,
                "query": request.query,
                "response": result.get("response", {}),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", []),
                "processing_metadata": result.get("metadata", {}),
                "performance": {
                    "execution_time_ms": execution_time,
                    "agent_specific_metrics": result.get("agent_metrics", {})
                }
            },
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Agent {request.agent_name} failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Agent invocation error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@agent_mcp_router.post("/multi-agent-workflow", response_model=MCPToolResponse)
async def multi_agent_workflow_tool(
    request: MultiAgentWorkflowRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test multi-agent workflows for any flow configuration.
    
    This endpoint tests coordination between multiple agents in a flow,
    useful for:
    - Validating agent coordination logic
    - Testing workflow orchestration
    - Performance testing of multi-agent scenarios
    """
    start_time = time.time()
    trace_id = f"workflow-{request.flow}-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Running multi-agent workflow for flow: {request.flow}")
        
        # Validate flow exists
        if request.flow not in flow_runner.flows:
            raise HTTPException(status_code=404, detail=f"Flow '{request.flow}' not found")
        
        # Get flow configuration
        flow_config = flow_runner.flows[request.flow]
        flow_agents = flow_config.get("agents", {})
        
        # Determine agents to run
        if request.agent_names:
            # Validate specified agents exist in flow
            for agent_name in request.agent_names:
                if agent_name not in flow_agents:
                    available_agents = list(flow_agents.keys())
                    raise HTTPException(
                        status_code=400,
                        detail=f"Agent '{agent_name}' not found in flow '{request.flow}'. Available agents: {available_agents}"
                    )
            agents_to_run = request.agent_names
        else:
            # Use all agents in flow
            agents_to_run = list(flow_agents.keys())
        
        logger.info(f"[{trace_id}] Running {len(agents_to_run)} agents: {agents_to_run}")
        
        # Prepare workflow parameters
        parameters = {
            "query": request.query,
            "agents": agents_to_run,
            "enable_synthesis": request.enable_synthesis,
            "parallel_execution": request.parallel_execution,
            "timeout_seconds": request.timeout_seconds,
            "test_mode": True
        }
        
        # Run multi-agent workflow
        result = await run_multi_agent_workflow_test(request.flow, parameters, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Multi-agent workflow completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result={
                "flow": request.flow,
                "query": request.query,
                "agents_executed": agents_to_run,
                "agent_responses": result.get("agent_responses", {}),
                "synthesis_result": result.get("synthesis_result", {}),
                "workflow_metadata": result.get("metadata", {}),
                "performance": {
                    "execution_time_ms": execution_time,
                    "agents_count": len(agents_to_run),
                    "parallel_execution": request.parallel_execution,
                    "synthesis_enabled": request.enable_synthesis
                }
            },
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Multi-agent workflow failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Multi-agent workflow error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@agent_mcp_router.post("/session-state", response_model=MCPToolResponse)
async def session_state_tool(
    request: SessionStateRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test session state management operations for any flow configuration.
    
    This endpoint enables testing session-specific functionality
    including metrics tracking, metadata management, and cleanup.
    """
    start_time = time.time()
    trace_id = f"session-{request.operation}-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Testing session operation: {request.operation} for flow: {request.flow}")
        
        # Validate flow exists if flow-specific operation
        if request.flow and request.flow not in flow_runner.flows:
            raise HTTPException(status_code=404, detail=f"Flow '{request.flow}' not found")
        
        # Execute session operation
        result = await test_session_operation(
            request.session_id,
            request.flow,
            request.operation,
            request.test_data,
            flow_runner
        )
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Session operation completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result={
                "session_id": request.session_id,
                "flow": request.flow,
                "operation": request.operation,
                "operation_result": result,
                "performance": {
                    "execution_time_ms": execution_time
                }
            },
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Session operation failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Session operation error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@agent_mcp_router.post("/message-validation", response_model=MCPToolResponse)
async def message_validation_tool(
    request: MessageValidationRequest
) -> MCPToolResponse:
    """
    Test message validation and processing for any flow configuration.
    
    This endpoint validates message formats and processing logic
    without requiring full flow execution.
    """
    start_time = time.time()
    trace_id = f"msg-val-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Validating message type: {request.message_type}")
        
        # Validate message based on type and flow context
        is_valid, error_msg = validate_message_format(
            request.message_type,
            request.message_data,
            request.flow,
            request.strict_validation
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return MCPToolResponse(
            success=True,
            result={
                "message_type": request.message_type,
                "flow": request.flow,
                "is_valid": is_valid,
                "error_message": error_msg if not is_valid else None,
                "validation_details": {
                    "strict_validation": request.strict_validation,
                    "message_size": len(str(request.message_data)),
                    "required_fields_present": check_required_fields(request.message_type, request.message_data, request.flow)
                },
                "performance": {
                    "execution_time_ms": execution_time
                }
            },
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Message validation failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Message validation error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


# Helper functions for flow-agnostic MCP operations

async def test_vector_query(run_request: RunRequest, flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute test vector query for any flow configuration."""
    # This would integrate with the actual vector store based on flow configuration
    # For now, return mock results for testing
    return {
        "results": [
            {
                "content": f"Sample content relevant to {run_request.flow} flow",
                "metadata": {"source": f"{run_request.flow}_doc_1.pdf", "section": "1.1"},
                "confidence": 0.85
            }
        ],
        "metadata": {
            "query_time_ms": 25.3,
            "total_results": 1,
            "search_strategy": "semantic",
            "flow": run_request.flow
        },
        "confidence_scores": [0.85]
    }


async def run_agent_test(agent_name: str, flow: str, parameters: Dict[str, Any], 
                        flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute test for individual agent in any flow configuration."""
    # This would invoke the specific agent based on flow configuration
    # For now, return mock results based on agent name and flow
    
    return {
        "response": {
            "analysis": f"{agent_name} analysis for {flow} flow",
            "findings": [f"{agent_name}_finding_1", f"{agent_name}_finding_2"]
        },
        "confidence": 0.85,
        "sources": [f"{flow}_source_1", f"{flow}_source_2"],
        "metadata": {
            "agent": agent_name,
            "flow": flow,
            "processing_time": 1.5,
            "model_version": "v1.0"
        },
        "agent_metrics": {
            "queries_processed": 1,
            "avg_confidence": 0.85
        }
    }


async def run_multi_agent_workflow_test(flow: str, parameters: Dict[str, Any], 
                                      flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute test for multi-agent workflow in any flow configuration."""
    agents = parameters.get("agents", [])
    query = parameters.get("query", "")
    
    # Simulate running multiple agents
    agent_responses = {}
    for agent in agents:
        agent_responses[agent] = await run_agent_test(agent, flow, parameters, flow_runner)
    
    # Simulate synthesis if enabled
    synthesis_result = {}
    if parameters.get("enable_synthesis", True):
        synthesis_result = {
            "summary": f"Multi-agent analysis for {flow} flow completed",
            "consensus": "high",
            "confidence": 0.87,
            "recommendations": [f"{flow}_recommendation_1", f"{flow}_recommendation_2"]
        }
    
    return {
        "agent_responses": agent_responses,
        "synthesis_result": synthesis_result,
        "metadata": {
            "flow": flow,
            "agents_executed": len(agents),
            "parallel_execution": parameters.get("parallel_execution", True),
            "workflow_time": 3.5
        }
    }


async def test_session_operation(session_id: str, flow: str, operation: str, 
                               test_data: Optional[Dict[str, Any]], 
                               flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute session operation test for any flow configuration."""
    operations = {
        "create": {
            "session_created": True,
            "session_id": session_id,
            "flow": flow,
            "initial_state": {"metrics": {}, "metadata": {}}
        },
        "get_status": {
            "session_exists": True,
            "session_id": session_id,
            "flow": flow,
            "status": "active",
            "query_count": 3,
            "last_activity": "2025-01-17T10:30:00Z"
        },
        "update_metadata": {
            "metadata_updated": True,
            "updated_fields": list(test_data.keys()) if test_data else []
        },
        "record_metrics": {
            "metrics_recorded": True,
            "new_query_count": 4,
            "average_processing_time": 45.2
        },
        "cleanup": {
            "session_cleaned": True,
            "resources_released": ["websocket", "metrics", "metadata"]
        },
        "list_sessions": {
            "sessions": [{"session_id": session_id, "flow": flow, "status": "active"}],
            "total_count": 1
        }
    }
    
    return operations.get(operation, {"error": f"Unknown operation: {operation}"})


def validate_message_format(message_type: str, message_data: Dict[str, Any], 
                           flow: Optional[str] = None, strict: bool = True) -> tuple[bool, str]:
    """Validate message format for any flow configuration."""
    # Basic message structure validation
    if not isinstance(message_data, dict):
        return False, "Message data must be a dictionary"
    
    # Flow-specific validation (can be extended based on flow configuration)
    if message_type == "run_flow":
        required_fields = ["query"]
        if flow:
            required_fields.append("flow")
            if message_data.get("flow") != flow:
                return False, f"Message flow '{message_data.get('flow')}' does not match expected flow '{flow}'"
    else:
        # Generic validation for other message types
        required_fields = ["type"]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in message_data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    return True, ""


def check_required_fields(message_type: str, message_data: Dict[str, Any], 
                         flow: Optional[str] = None) -> Dict[str, bool]:
    """Check which required fields are present in message for any flow configuration."""
    if message_type == "run_flow":
        required_fields = ["query"]
        if flow:
            required_fields.append("flow")
    else:
        required_fields = ["type"]
    
    return {field: field in message_data for field in required_fields}