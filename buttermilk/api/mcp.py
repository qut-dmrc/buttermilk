"""MCP (Model Context Protocol) tool endpoints for Buttermilk.

This module provides MCP-compliant HTTP endpoints that expose Buttermilk agent capabilities
as tools that can be called by MCP clients or used directly via HTTP. These endpoints
run alongside the existing WebSocket infrastructure without breaking changes.

Key features:
- Stateless tool calls for simple operations
- Structured error handling and responses
- Direct agent invocation without complex session management
- MCP protocol compliance for external client integration
"""

import asyncio
import traceback
from typing import Any, Dict

import shortuuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from buttermilk._core import logger
from buttermilk._core.types import RunRequest
from buttermilk.api.routes import get_flows
from buttermilk.runner.flowrunner import FlowRunner

# Create the MCP router
mcp_router = APIRouter(prefix="/mcp", tags=["MCP Tools"])


class MCPToolResponse(BaseModel):
    """Standardized response format for MCP tool calls."""
    
    success: bool = Field(..., description="Whether the tool call succeeded")
    result: Any | None = Field(None, description="Tool result data if successful")
    error: str | None = Field(None, description="Error message if failed")
    trace_id: str | None = Field(None, description="Trace ID for debugging")
    execution_time_ms: float | None = Field(None, description="Execution time in milliseconds")


class MCPError(Exception):
    """Custom exception for MCP tool errors."""
    
    def __init__(self, message: str, details: Dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


async def run_single_agent(
    agent_role: str,
    flow_name: str,
    parameters: Dict[str, Any],
    flow_runner: FlowRunner
) -> Any:
    """Helper function to run a single agent in isolation.
    
    Args:
        agent_role: The role of the agent to run (e.g., 'judge', 'synth')
        flow_name: The flow configuration to use
        parameters: Parameters to pass to the agent
        flow_runner: The FlowRunner instance
        
    Returns:
        The agent's output result
        
    Raises:
        MCPError: If agent execution fails
    """
    if flow_name not in flow_runner.flows:
        raise MCPError(f"Flow '{flow_name}' not found. Available flows: {list(flow_runner.flows.keys())}")
    
    flow_config = flow_runner.flows[flow_name]
    
    # Check if the requested agent role exists in the flow
    if agent_role not in flow_config.agents:
        available_agents = list(flow_config.agents.keys())
        raise MCPError(f"Agent role '{agent_role}' not found in flow '{flow_name}'. Available agents: {available_agents}")
    
    # Create a minimal RunRequest for this tool call
    session_id = f"mcp-{shortuuid.uuid()[:8]}"
    run_request = RunRequest(
        ui_type="mcp",
        flow=flow_name,
        session_id=session_id,
        parameters=parameters,
        callback_to_ui=None  # No UI callback for MCP calls
    )
    
    try:
        # Get the agent configuration
        agent_config = flow_config.agents[agent_role]
        
        # Get the first agent class and config (simplified for single agent execution)
        agent_configs = list(agent_config.get_configs(params=run_request, flow_default_params=flow_config.parameters))
        if not agent_configs:
            raise MCPError(f"No agent configurations found for role '{agent_role}'")
        
        agent_cls, variant_config = agent_configs[0]
        
        # Create agent instance with session ID
        config_with_session = {**variant_config.model_dump(), "session_id": session_id}
        agent_instance = agent_cls(**config_with_session)
        
        # Create AgentInput from parameters
        from buttermilk._core.agent import AgentInput
        agent_input = AgentInput(inputs=parameters)
        
        # Invoke the agent
        agent_trace = await agent_instance.invoke(agent_input)
        
        if not agent_trace.outputs:
            raise MCPError(f"Agent '{agent_role}' produced no output")
        
        return agent_trace.outputs
        
    except Exception as e:
        if isinstance(e, MCPError):
            raise
        logger.error(f"Error running agent '{agent_role}': {e}")
        raise MCPError(f"Agent execution failed: {str(e)}", {"traceback": traceback.format_exc()})


class JudgeRequest(BaseModel):
    """Request model for judge tool."""
    text: str = Field(..., description="Text content to judge")
    criteria: str = Field(..., description="Criteria to judge against") 
    model: str = Field("gpt4o", description="LLM model to use")
    flow: str = Field("tox", description="Flow configuration to use")

@mcp_router.post("/tools/judge", response_model=MCPToolResponse)
async def judge_content(
    request: JudgeRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """Judge content against specified criteria using the judge agent.
    
    This endpoint runs the judge agent in isolation to evaluate content
    against specific criteria. It's designed for simple, stateless operations.
    
    Args:
        text: The text content to be judged
        criteria: The criteria to judge the content against
        model: The LLM model to use (default: gpt4o)
        flow: The flow configuration to use (default: tox)
        
    Returns:
        MCPToolResponse with judge results or error information
    """
    import time
    start_time = time.time()
    trace_id = f"judge-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Judge tool called with criteria: {request.criteria}")
        
        # Prepare parameters for the judge agent
        parameters = {
            "text": request.text,
            "criteria": request.criteria,
            "model": request.model
        }
        
        # Run the judge agent
        result = await run_single_agent("judge", request.flow, parameters, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Judge tool completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result=result.model_dump() if hasattr(result, 'model_dump') else result,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except MCPError as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Judge tool failed: {e.message}")
        
        return MCPToolResponse(
            success=False,
            error=e.message,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Judge tool error: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


class SynthesizeRequest(BaseModel):
    """Request model for synthesize tool."""
    text: str = Field(..., description="Text content to synthesize")
    criteria: str = Field(..., description="Criteria to use for synthesis")
    model: str = Field("gpt4o", description="LLM model to use")
    flow: str = Field("tox", description="Flow configuration to use")

@mcp_router.post("/tools/synthesize", response_model=MCPToolResponse)
async def synthesize_content(
    request: SynthesizeRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """Synthesize content based on specified criteria using the synth agent.
    
    This endpoint runs the synth agent in isolation to create synthesized
    content based on input text and criteria.
    
    Args:
        text: The text content to synthesize
        criteria: The criteria to use for synthesis
        model: The LLM model to use (default: gpt4o)
        flow: The flow configuration to use (default: tox)
        
    Returns:
        MCPToolResponse with synthesis results or error information
    """
    import time
    start_time = time.time()
    trace_id = f"synth-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Synthesize tool called with criteria: {request.criteria}")
        
        # Prepare parameters for the synth agent
        parameters = {
            "text": request.text,
            "criteria": request.criteria,
            "model": request.model
        }
        
        # Run the synth agent
        result = await run_single_agent("synth", request.flow, parameters, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Synthesize tool completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result=result.model_dump() if hasattr(result, 'model_dump') else result,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except MCPError as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Synthesize tool failed: {e.message}")
        
        return MCPToolResponse(
            success=False,
            error=e.message,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Synthesize tool error: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


class DifferencesRequest(BaseModel):
    """Request model for differences tool."""
    text1: str = Field(..., description="First text to compare")
    text2: str = Field(..., description="Second text to compare")
    criteria: str = Field(..., description="Criteria to use for comparison")
    model: str = Field("gpt4o", description="LLM model to use")
    flow: str = Field("tox", description="Flow configuration to use")

@mcp_router.post("/tools/find_differences", response_model=MCPToolResponse)
async def find_differences(
    request: DifferencesRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """Find differences between two texts using the differences agent.
    
    This endpoint runs the differences agent in isolation to identify
    differences between two pieces of text based on specified criteria.
    
    Args:
        text1: The first text to compare
        text2: The second text to compare
        criteria: The criteria to use for comparison
        model: The LLM model to use (default: gpt4o)
        flow: The flow configuration to use (default: tox)
        
    Returns:
        MCPToolResponse with differences analysis or error information
    """
    import time
    start_time = time.time()
    trace_id = f"diff-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Differences tool called with criteria: {request.criteria}")
        
        # Prepare parameters for the differences agent
        parameters = {
            "text1": request.text1,
            "text2": request.text2,
            "criteria": request.criteria,
            "model": request.model
        }
        
        # Run the differences agent
        result = await run_single_agent("differences", request.flow, parameters, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Differences tool completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result=result.model_dump() if hasattr(result, 'model_dump') else result,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except MCPError as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Differences tool failed: {e.message}")
        
        return MCPToolResponse(
            success=False,
            error=e.message,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Differences tool error: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@mcp_router.get("/tools", response_model=MCPToolResponse)
async def list_tools() -> MCPToolResponse:
    """List all available MCP tools.
    
    Returns:
        MCPToolResponse with list of available tools and their descriptions
    """
    tools = [
        {
            "name": "judge",
            "endpoint": "/mcp/tools/judge",
            "description": "Judge content against specified criteria",
            "parameters": ["text", "criteria", "model", "flow"]
        },
        {
            "name": "synthesize", 
            "endpoint": "/mcp/tools/synthesize",
            "description": "Synthesize content based on criteria",
            "parameters": ["text", "criteria", "model", "flow"]
        },
        {
            "name": "find_differences",
            "endpoint": "/mcp/tools/find_differences", 
            "description": "Find differences between two texts",
            "parameters": ["text1", "text2", "criteria", "model", "flow"]
        }
    ]
    
    return MCPToolResponse(
        success=True,
        result={"tools": tools, "total": len(tools)}
    )