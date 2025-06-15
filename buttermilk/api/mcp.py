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


class FlowStartRequest(BaseModel):
    """Request model for starting interactive flows."""
    flow: str = Field(..., description="Flow name to start")
    record_id: str = Field("", description="Optional record ID to analyze")
    prompt: str = Field("", description="Optional prompt for the flow")
    uri: str = Field("", description="Optional URI to analyze")
    session_id: str | None = Field(None, description="Optional session ID for tracking")


@mcp_router.post("/flows/start", response_model=MCPToolResponse)
async def start_flow(
    request: FlowStartRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """Start an interactive flow with optional session tracking.
    
    This endpoint initiates a complete Buttermilk flow that can include
    multiple agents working together through the orchestration system.
    Unlike the simple tool endpoints, this provides full flow capabilities.
    
    Args:
        flow: The flow name to execute
        record_id: Optional specific record to analyze
        prompt: Optional prompt to provide to the flow
        uri: Optional URI for the flow to analyze
        session_id: Optional session ID for tracking (auto-generated if not provided)
        
    Returns:
        MCPToolResponse with flow execution results or error information
    """
    import time
    start_time = time.time()
    
    # Generate session ID if not provided
    session_id = request.session_id or f"mcp-flow-{shortuuid.uuid()[:8]}"
    trace_id = f"flow-{session_id}"
    
    try:
        logger.info(f"[{trace_id}] Starting flow '{request.flow}' with session {session_id}")
        
        # Validate flow exists
        if request.flow not in flow_runner.flows:
            available_flows = list(flow_runner.flows.keys())
            raise MCPError(f"Flow '{request.flow}' not found. Available flows: {available_flows}")
        
        # Create RunRequest for the flow
        run_request = RunRequest(
            ui_type="mcp",
            flow=request.flow,
            record_id=request.record_id,
            prompt=request.prompt,
            uri=request.uri,
            session_id=session_id,
            callback_to_ui=None  # No UI callback for MCP calls
        )
        
        # Start the flow execution
        logger.info(f"[{trace_id}] Executing flow with parameters: record_id={request.record_id}, prompt={request.prompt[:50] if request.prompt else 'None'}...")
        
        # Run the flow and wait for completion
        flow_result = await flow_runner.run_flow(
            run_request=run_request,
            wait_for_completion=True
        )
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] Flow '{request.flow}' completed in {execution_time:.2f}ms")
        
        # Extract meaningful results from flow execution
        result_data = {
            "session_id": session_id,
            "flow": request.flow,
            "status": "completed",
            "execution_time_ms": execution_time
        }
        
        # Try to extract agent results if available
        if hasattr(flow_result, 'outputs') and flow_result.outputs:
            result_data["outputs"] = flow_result.outputs
        elif hasattr(flow_result, 'model_dump'):
            result_data["flow_result"] = flow_result.model_dump()
        else:
            result_data["flow_result"] = str(flow_result)
        
        return MCPToolResponse(
            success=True,
            result=result_data,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except MCPError as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Flow start failed: {e.message}")
        
        return MCPToolResponse(
            success=False,
            error=e.message,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Flow start error: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Flow execution failed: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


class AnalyzeRecordRequest(BaseModel):
    """Request model for multi-agent record analysis."""
    record_id: str = Field(..., description="Record ID to analyze")
    agents: list[str] = Field(..., description="List of agent roles to run")
    flow: str = Field("tox", description="Flow configuration to use")
    criteria: str = Field("", description="Optional criteria for analysis")
    model: str = Field("gpt4o", description="LLM model to use")


@mcp_router.post("/tools/analyze_record", response_model=MCPToolResponse)
async def analyze_record(
    request: AnalyzeRecordRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """Analyze a record using multiple agents in parallel.
    
    This endpoint runs multiple agents against the same record content
    to provide comprehensive analysis from different perspectives.
    Agents run in parallel for efficiency.
    
    Args:
        record_id: The record ID to analyze
        agents: List of agent roles to run (e.g., ['judge', 'synth'])
        flow: The flow configuration to use (default: tox)
        criteria: Optional criteria for all agents
        model: The LLM model to use (default: gpt4o)
        
    Returns:
        MCPToolResponse with analysis results from all agents
    """
    import time
    start_time = time.time()
    trace_id = f"analyze-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Analyzing record '{request.record_id}' with agents: {request.agents}")
        
        # Validate flow exists
        if request.flow not in flow_runner.flows:
            available_flows = list(flow_runner.flows.keys())
            raise MCPError(f"Flow '{request.flow}' not found. Available flows: {available_flows}")
        
        flow_config = flow_runner.flows[request.flow]
        
        # Validate all requested agents exist in the flow
        missing_agents = [agent for agent in request.agents if agent not in flow_config.agents]
        if missing_agents:
            available_agents = list(flow_config.agents.keys())
            raise MCPError(f"Agents {missing_agents} not found in flow '{request.flow}'. Available agents: {available_agents}")
        
        # Get record content - we'll need to load it from the flow's storage
        # For now, we'll pass the record_id to each agent and let them handle loading
        record_data = {"record_id": request.record_id}
        if request.criteria:
            record_data["criteria"] = request.criteria
        record_data["model"] = request.model
        
        # Run all agents in parallel
        logger.info(f"[{trace_id}] Running {len(request.agents)} agents in parallel")
        
        async def run_agent_task(agent_role: str):
            """Run a single agent and return its result with role info."""
            try:
                result = await run_single_agent(agent_role, request.flow, record_data, flow_runner)
                return {
                    "agent": agent_role,
                    "success": True,
                    "result": result.model_dump() if hasattr(result, 'model_dump') else result
                }
            except Exception as e:
                logger.error(f"[{trace_id}] Agent '{agent_role}' failed: {e}")
                return {
                    "agent": agent_role,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute all agents concurrently
        agent_tasks = [run_agent_task(agent) for agent in request.agents]
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in agent_results:
            if isinstance(result, Exception):
                failed_results.append({
                    "agent": "unknown",
                    "success": False,
                    "error": str(result)
                })
            elif result.get("success"):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Prepare comprehensive result
        analysis_result = {
            "record_id": request.record_id,
            "agents_requested": request.agents,
            "successful_analyses": len(successful_results),
            "failed_analyses": len(failed_results),
            "execution_time_ms": execution_time,
            "results": successful_results
        }
        
        if failed_results:
            analysis_result["failures"] = failed_results
        
        # Determine overall success
        overall_success = len(successful_results) > 0
        
        if overall_success:
            logger.info(f"[{trace_id}] Record analysis completed: {len(successful_results)}/{len(request.agents)} agents succeeded")
        else:
            logger.error(f"[{trace_id}] Record analysis failed: all agents failed")
        
        return MCPToolResponse(
            success=overall_success,
            result=analysis_result,
            trace_id=trace_id,
            execution_time_ms=execution_time,
            error=f"Some agents failed: {[f['agent'] for f in failed_results]}" if failed_results and overall_success else None
        )
        
    except MCPError as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Record analysis failed: {e.message}")
        
        return MCPToolResponse(
            success=False,
            error=e.message,
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] Record analysis error: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Analysis failed: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@mcp_router.get("/debug/flow/{flow_name}", response_model=MCPToolResponse)
async def debug_flow_config(
    flow_name: str,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """Debug endpoint to inspect flow configuration structure."""
    try:
        if flow_name not in flow_runner.flows:
            available_flows = list(flow_runner.flows.keys())
            raise MCPError(f"Flow '{flow_name}' not found. Available flows: {available_flows}")
        
        flow_config = flow_runner.flows[flow_name]
        
        # Inspect the structure
        debug_info = {
            "flow_name": flow_name,
            "flow_type": str(type(flow_config)),
            "agents_type": str(type(flow_config.agents)) if hasattr(flow_config, 'agents') else "No agents attribute",
            "agents_keys": list(flow_config.agents.keys()) if hasattr(flow_config, 'agents') and flow_config.agents else [],
            "agent_details": {}
        }
        
        if hasattr(flow_config, 'agents') and flow_config.agents:
            for agent_name, agent_config in flow_config.agents.items():
                debug_info["agent_details"][agent_name] = {
                    "type": str(type(agent_config)),
                    "has_get_configs": hasattr(agent_config, 'get_configs'),
                    "dir": [attr for attr in dir(agent_config) if not attr.startswith('_')][:10]  # First 10 methods
                }
        
        return MCPToolResponse(success=True, result=debug_info)
        
    except Exception as e:
        return MCPToolResponse(success=False, error=str(e))


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
        },
        {
            "name": "analyze_record",
            "endpoint": "/mcp/tools/analyze_record",
            "description": "Analyze a record using multiple agents in parallel",
            "parameters": ["record_id", "agents", "flow", "criteria", "model"]
        }
    ]
    
    flows = [
        {
            "name": "start_flow",
            "endpoint": "/mcp/flows/start",
            "description": "Start an interactive flow with full orchestration",
            "parameters": ["flow", "record_id", "prompt", "uri", "session_id"]
        }
    ]
    
    return MCPToolResponse(
        success=True,
        result={
            "tools": tools, 
            "flows": flows,
            "total_tools": len(tools),
            "total_flows": len(flows)
        }
    )