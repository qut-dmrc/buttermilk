"""OSB MCP Endpoints for Individual Component Testing.

This module provides MCP-compliant HTTP endpoints specifically for testing
OSB (Oversight Board) components in isolation. These endpoints enable:

- Individual OSB agent testing (researcher, policy_analyst, fact_checker, explorer)
- Vector store query testing without full flow execution
- Session state management testing
- Multi-agent response synthesis testing
- OSB message validation and processing testing

All endpoints follow MCP protocol standards and integrate with existing
test infrastructure for comprehensive OSB validation.
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

# Create the OSB MCP router
osb_mcp_router = APIRouter(prefix="/mcp/osb", tags=["OSB MCP Tools"])


# Request/Response Models for OSB MCP Endpoints

class OSBVectorQueryRequest(BaseModel):
    """Request model for OSB vector store queries."""
    
    query: str = Field(..., description="Query text for vector search")
    flow: str = Field(default="osb", description="Flow configuration to use")
    agent_type: str = Field(default="researcher", description="Type of OSB agent for query context")
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
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        valid_types = ["researcher", "policy_analyst", "fact_checker", "explorer"]
        if v not in valid_types:
            raise ValueError(f"Agent type must be one of: {valid_types}")
        return v


class OSBAgentInvokeRequest(BaseModel):
    """Request model for individual OSB agent invocation."""
    
    query: str = Field(..., description="Policy analysis query")
    agent_name: str = Field(..., description="Specific OSB agent to invoke")
    flow: str = Field(default="osb", description="Flow configuration to use")
    
    # OSB-specific context
    case_number: Optional[str] = Field(None, description="OSB case identifier")
    content_type: Optional[str] = Field(None, description="Type of content being analyzed")
    platform: Optional[str] = Field(None, description="Platform where content originated")
    
    # Agent-specific parameters
    enable_precedent_analysis: bool = Field(default=True, description="Include precedent analysis")
    include_policy_references: bool = Field(default=True, description="Include policy references")
    max_processing_time: int = Field(default=30, description="Maximum processing time in seconds")
    
    @validator('agent_name')
    def validate_agent_name(cls, v):
        valid_agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]
        if v not in valid_agents:
            raise ValueError(f"Agent name must be one of: {valid_agents}")
        return v


class OSBSynthesisRequest(BaseModel):
    """Request model for multi-agent response synthesis."""
    
    query: str = Field(..., description="Original OSB query")
    agent_responses: Dict[str, Any] = Field(..., description="Individual agent responses to synthesize")
    flow: str = Field(default="osb", description="Flow configuration to use")
    
    # Synthesis parameters
    enable_cross_validation: bool = Field(default=True, description="Enable cross-validation between agents")
    confidence_weighting: bool = Field(default=True, description="Weight synthesis by agent confidence scores")
    include_precedents: bool = Field(default=True, description="Include precedent case analysis")
    
    @validator('agent_responses')
    def validate_agent_responses(cls, v):
        if not v:
            raise ValueError("Agent responses cannot be empty")
        
        valid_agents = {"researcher", "policy_analyst", "fact_checker", "explorer"}
        for agent_name in v.keys():
            if agent_name not in valid_agents:
                raise ValueError(f"Invalid agent name: {agent_name}")
        
        return v


class OSBSessionStateRequest(BaseModel):
    """Request model for OSB session state testing."""
    
    session_id: str = Field(..., description="Session identifier to test")
    operation: str = Field(..., description="Session operation to perform")
    test_data: Optional[Dict[str, Any]] = Field(None, description="Test data for session operations")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_ops = ["create", "get_status", "update_metadata", "record_metrics", "cleanup"]
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class OSBMessageValidationRequest(BaseModel):
    """Request model for OSB message validation testing."""
    
    message_type: str = Field(..., description="Type of OSB message to validate")
    message_data: Dict[str, Any] = Field(..., description="Message data to validate")
    strict_validation: bool = Field(default=True, description="Enable strict validation rules")
    
    @validator('message_type')
    def validate_message_type(cls, v):
        valid_types = ["osb_query", "osb_status", "osb_partial", "osb_complete", "osb_error"]
        if v not in valid_types:
            raise ValueError(f"Message type must be one of: {valid_types}")
        return v


# OSB MCP Endpoint Implementations

@osb_mcp_router.post("/vector-query", response_model=MCPToolResponse)
async def osb_vector_query_tool(
    request: OSBVectorQueryRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test OSB vector store queries without full flow execution.
    
    This endpoint allows testing individual vector store queries
    with OSB-specific context and parameters, useful for:
    - Validating vector store connectivity
    - Testing query performance and results
    - Debugging search strategies
    """
    start_time = time.time()
    trace_id = f"osb-vector-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] OSB vector query: {request.query[:100]}...")
        
        # Prepare parameters for vector query
        parameters = {
            "query": request.query,
            "agent_type": request.agent_type,
            "max_results": request.max_results,
            "confidence_threshold": request.confidence_threshold,
            "include_metadata": request.include_metadata,
            "test_mode": True  # Flag to indicate this is a test query
        }
        
        # Create a test session for the vector query
        session_id = f"test-vector-{trace_id}"
        
        # Use the OSB flow with vector query parameters
        run_request = RunRequest(
            ui_type="mcp",
            flow=request.flow,
            session_id=session_id,
            parameters=parameters
        )
        
        # Execute vector query (this would need implementation in OSB agents)
        result = await test_osb_vector_query(run_request, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] OSB vector query completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result={
                "query": request.query,
                "agent_type": request.agent_type,
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
        logger.error(f"[{trace_id}] OSB vector query failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Vector query error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@osb_mcp_router.post("/agent-invoke", response_model=MCPToolResponse)
async def osb_agent_invoke_tool(
    request: OSBAgentInvokeRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test individual OSB agent invocations in isolation.
    
    This endpoint enables testing specific OSB agents without
    running the full multi-agent workflow, useful for:
    - Agent-specific functionality validation
    - Performance testing per agent
    - Debugging individual agent responses
    """
    start_time = time.time()
    trace_id = f"osb-agent-{request.agent_name}-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Invoking OSB agent: {request.agent_name}")
        
        # Prepare OSB-specific parameters
        parameters = {
            "osb_query": request.query,
            "case_number": request.case_number,
            "content_type": request.content_type,
            "platform": request.platform,
            "enable_precedent_analysis": request.enable_precedent_analysis,
            "include_policy_references": request.include_policy_references,
            "max_processing_time": request.max_processing_time,
            "agent_name": request.agent_name,
            "test_mode": True
        }
        
        # Run the specific OSB agent
        result = await run_osb_agent_test(request.agent_name, request.flow, parameters, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] OSB agent {request.agent_name} completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result={
                "agent_name": request.agent_name,
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
        logger.error(f"[{trace_id}] OSB agent {request.agent_name} failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Agent invocation error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@osb_mcp_router.post("/synthesis", response_model=MCPToolResponse)
async def osb_synthesis_tool(
    request: OSBSynthesisRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test multi-agent response synthesis for OSB workflows.
    
    This endpoint tests the synthesis of multiple OSB agent responses
    into a coherent analysis, useful for:
    - Validating synthesis logic
    - Testing cross-validation between agents
    - Performance testing of synthesis algorithms
    """
    start_time = time.time()
    trace_id = f"osb-synthesis-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Synthesizing {len(request.agent_responses)} agent responses")
        
        # Prepare synthesis parameters
        parameters = {
            "original_query": request.query,
            "agent_responses": request.agent_responses,
            "enable_cross_validation": request.enable_cross_validation,
            "confidence_weighting": request.confidence_weighting,
            "include_precedents": request.include_precedents,
            "test_mode": True
        }
        
        # Run synthesis process
        result = await run_osb_synthesis_test(request.flow, parameters, flow_runner)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"[{trace_id}] OSB synthesis completed in {execution_time:.2f}ms")
        
        return MCPToolResponse(
            success=True,
            result={
                "synthesis_summary": result.get("synthesis_summary", ""),
                "confidence_score": result.get("confidence_score", 0.0),
                "policy_violations": result.get("policy_violations", []),
                "recommendations": result.get("recommendations", []),
                "precedent_cases": result.get("precedent_cases", []),
                "cross_validation_results": result.get("cross_validation", {}),
                "synthesis_metadata": result.get("metadata", {}),
                "performance": {
                    "execution_time_ms": execution_time,
                    "agents_processed": len(request.agent_responses),
                    "validation_score": result.get("validation_score", 0.0)
                }
            },
            trace_id=trace_id,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{trace_id}] OSB synthesis failed: {e}")
        
        return MCPToolResponse(
            success=False,
            error=f"Synthesis error: {str(e)}",
            trace_id=trace_id,
            execution_time_ms=execution_time
        )


@osb_mcp_router.post("/session-state", response_model=MCPToolResponse)
async def osb_session_state_tool(
    request: OSBSessionStateRequest,
    flow_runner: FlowRunner = Depends(get_flows)
) -> MCPToolResponse:
    """
    Test OSB session state management operations.
    
    This endpoint enables testing session-specific functionality
    including metrics tracking, metadata management, and cleanup.
    """
    start_time = time.time()
    trace_id = f"osb-session-{request.operation}-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Testing OSB session operation: {request.operation}")
        
        # Execute session operation
        result = await test_osb_session_operation(
            request.session_id,
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


@osb_mcp_router.post("/message-validation", response_model=MCPToolResponse)
async def osb_message_validation_tool(
    request: OSBMessageValidationRequest
) -> MCPToolResponse:
    """
    Test OSB message validation and processing.
    
    This endpoint validates OSB-specific message formats and
    processing logic without requiring full flow execution.
    """
    start_time = time.time()
    trace_id = f"osb-msg-val-{shortuuid.uuid()[:8]}"
    
    try:
        logger.info(f"[{trace_id}] Validating OSB message type: {request.message_type}")
        
        # Import OSB message processor
        from buttermilk.api.osb_message_enhancements import OSBMessageProcessor
        
        # Validate message based on type
        if request.message_type == "osb_query":
            is_valid, error_msg = OSBMessageProcessor.validate_osb_message({
                "type": "run_flow",
                "flow": "osb",
                **request.message_data
            })
        else:
            # Validate other OSB message types
            is_valid, error_msg = validate_osb_message_type(
                request.message_type,
                request.message_data,
                request.strict_validation
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        return MCPToolResponse(
            success=True,
            result={
                "message_type": request.message_type,
                "is_valid": is_valid,
                "error_message": error_msg if not is_valid else None,
                "validation_details": {
                    "strict_validation": request.strict_validation,
                    "message_size": len(str(request.message_data)),
                    "required_fields_present": check_required_fields(request.message_type, request.message_data)
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


# Helper functions for OSB MCP operations

async def test_osb_vector_query(run_request: RunRequest, flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute test vector query for OSB."""
    # This would integrate with the actual OSB vector store
    # For now, return mock results for testing
    return {
        "results": [
            {
                "content": "Sample policy document content",
                "metadata": {"source": "policy_doc_1.pdf", "section": "4.2"},
                "confidence": 0.85
            }
        ],
        "metadata": {
            "query_time_ms": 25.3,
            "total_results": 1,
            "search_strategy": "semantic"
        },
        "confidence_scores": [0.85]
    }


async def run_osb_agent_test(agent_name: str, flow: str, parameters: Dict[str, Any], 
                           flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute test for individual OSB agent."""
    # This would invoke the specific OSB agent
    # For now, return mock results based on agent type
    
    agent_responses = {
        "researcher": {
            "response": {"findings": "Content analysis completed", "evidence": ["source1", "source2"]},
            "confidence": 0.85,
            "sources": ["policy_doc_1.pdf", "precedent_case_123"],
            "metadata": {"search_time": 1.2, "sources_found": 5}
        },
        "policy_analyst": {
            "response": {"analysis": "Policy violation detected", "recommendations": ["action1", "action2"]},
            "confidence": 0.90,
            "sources": ["community_standards.pdf"],
            "metadata": {"analysis_time": 2.1, "policies_checked": 12}
        },
        "fact_checker": {
            "response": {"validation": "Claims verified", "accuracy_score": 0.92},
            "confidence": 0.88,
            "sources": ["official_sources.pdf"],
            "metadata": {"validation_time": 1.8, "facts_checked": 8}
        },
        "explorer": {
            "response": {"themes": ["hate_speech", "harassment"], "related_cases": ["OSB-123", "OSB-456"]},
            "confidence": 0.82,
            "sources": ["case_database"],
            "metadata": {"exploration_time": 2.5, "themes_found": 15}
        }
    }
    
    return agent_responses.get(agent_name, {
        "response": {"error": f"Unknown agent: {agent_name}"},
        "confidence": 0.0,
        "sources": [],
        "metadata": {}
    })


async def run_osb_synthesis_test(flow: str, parameters: Dict[str, Any], 
                               flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute test for OSB response synthesis."""
    # This would implement the actual synthesis logic
    # For now, return mock synthesis results
    
    return {
        "synthesis_summary": "Multi-agent analysis indicates policy violations with high confidence",
        "confidence_score": 0.87,
        "policy_violations": ["Hate speech (Section 4.2)", "Harassment (Section 3.1)"],
        "recommendations": ["Remove content", "Issue warning", "Monitor user"],
        "precedent_cases": ["OSB-2024-089", "OSB-2024-156"],
        "cross_validation": {
            "researcher_policy_analyst_agreement": 0.92,
            "fact_checker_validation_score": 0.88,
            "explorer_theme_consistency": 0.85
        },
        "metadata": {
            "synthesis_time": 0.8,
            "agents_synthesized": len(parameters.get("agent_responses", {})),
            "validation_passes": 3
        },
        "validation_score": 0.89
    }


async def test_osb_session_operation(session_id: str, operation: str, 
                                   test_data: Optional[Dict[str, Any]], 
                                   flow_runner: FlowRunner) -> Dict[str, Any]:
    """Execute OSB session operation test."""
    # This would integrate with the actual session management
    # For now, return mock results based on operation
    
    operations = {
        "create": {
            "session_created": True,
            "session_id": session_id,
            "initial_state": {"osb_metrics": {}, "case_metadata": {}}
        },
        "get_status": {
            "session_exists": True,
            "session_id": session_id,
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
        }
    }
    
    return operations.get(operation, {"error": f"Unknown operation: {operation}"})


def validate_osb_message_type(message_type: str, message_data: Dict[str, Any], 
                             strict: bool = True) -> tuple[bool, str]:
    """Validate OSB message format."""
    required_fields = {
        "osb_status": ["session_id", "status"],
        "osb_partial": ["session_id", "agent", "partial_response"],
        "osb_complete": ["session_id", "synthesis_summary", "agent_responses"],
        "osb_error": ["session_id", "error_type", "error_message"]
    }
    
    if message_type not in required_fields:
        return False, f"Unknown message type: {message_type}"
    
    missing_fields = []
    for field in required_fields[message_type]:
        if field not in message_data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    return True, ""


def check_required_fields(message_type: str, message_data: Dict[str, Any]) -> Dict[str, bool]:
    """Check which required fields are present in message."""
    required_fields = {
        "osb_query": ["query", "flow"],
        "osb_status": ["session_id", "status"],
        "osb_partial": ["session_id", "agent", "partial_response"],
        "osb_complete": ["session_id", "synthesis_summary", "agent_responses"],
        "osb_error": ["session_id", "error_type", "error_message"]
    }
    
    fields = required_fields.get(message_type, [])
    return {field: field in message_data for field in fields}