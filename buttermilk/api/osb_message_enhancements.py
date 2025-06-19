"""
OSB Message Processing Enhancements

Extensions to the existing message processing pipeline to support
OSB-specific message types and enhanced functionality. 

This module integrates with:
- buttermilk.api.services.message_service for WebSocket message processing
- buttermilk.runner.flowrunner for session management  
- buttermilk.runner.osb_session_enhancements for OSB session features

Key principles:
- Extend existing infrastructure rather than replace it
- Maintain backward compatibility with current message types
- Add OSB-specific message validation and processing
"""

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, validator

from buttermilk._core.log import logger
from buttermilk._core.config import RunRequest
from buttermilk._core.contract import UIMessage, FlowMessage
from buttermilk.runner.flowrunner import FlowRunContext


class OSBQueryMessage(BaseModel):
    """Enhanced message structure for OSB queries."""
    
    type: str = Field(default="run_flow", description="Message type")
    flow: str = Field(default="osb", description="Flow name")
    query: str = Field(..., description="OSB policy analysis query")
    
    # OSB-specific metadata
    case_number: Optional[str] = Field(None, description="OSB case identifier")
    case_priority: str = Field(default="medium", description="Case priority level")
    content_type: Optional[str] = Field(None, description="Type of content being analyzed")
    platform: Optional[str] = Field(None, description="Platform where content originated")
    user_context: Optional[str] = Field(None, description="Context about the user/account")
    
    # OSB processing options
    enable_multi_agent_synthesis: bool = Field(default=True, description="Enable multi-agent analysis")
    enable_cross_validation: bool = Field(default=True, description="Enable fact-checker validation")
    enable_precedent_analysis: bool = Field(default=True, description="Include precedent case analysis")
    include_policy_references: bool = Field(default=True, description="Include policy document references")
    
    # Performance options
    enable_streaming_response: bool = Field(default=True, description="Stream partial responses")
    max_processing_time: int = Field(default=60, description="Maximum processing time in seconds")
    
    @validator('query')
    def validate_query_length(cls, v):
        """Validate query length according to OSB configuration."""
        if len(v) > 2000:  # From osb.yaml max_query_length
            raise ValueError("Query exceeds maximum length of 2000 characters")
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator('case_priority')
    def validate_priority(cls, v):
        """Validate case priority levels."""
        valid_priorities = ["low", "medium", "high", "critical"]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v


class OSBStatusMessage(BaseModel):
    """Status update message for OSB processing."""
    
    type: str = Field(default="osb_status", description="Message type")
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Current processing status")
    agent: Optional[str] = Field(None, description="Current agent being processed")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage (0-100)")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    message: Optional[str] = Field(None, description="Human-readable status message")


class OSBPartialResponseMessage(BaseModel):
    """Partial response message for streaming OSB results."""
    
    type: str = Field(default="osb_partial", description="Message type")
    session_id: str = Field(..., description="Session identifier")
    agent: str = Field(..., description="Agent providing partial response")
    partial_response: str = Field(..., description="Partial response content")
    confidence: Optional[float] = Field(None, description="Confidence score for partial response")
    sources: Optional[list[str]] = Field(None, description="Sources referenced in partial response")


class OSBCompleteResponseMessage(BaseModel):
    """Complete response message for OSB analysis."""
    
    type: str = Field(default="osb_complete", description="Message type")
    session_id: str = Field(..., description="Session identifier")
    
    # Analysis results
    synthesis_summary: str = Field(..., description="Overall synthesis of multi-agent analysis")
    agent_responses: Dict[str, Any] = Field(..., description="Individual agent responses")
    
    # OSB-specific results
    policy_violations: list[str] = Field(default_factory=list, description="Identified policy violations")
    recommendations: list[str] = Field(default_factory=list, description="Recommended actions")
    precedent_cases: list[str] = Field(default_factory=list, description="Related precedent cases")
    confidence_score: float = Field(..., description="Overall confidence in analysis")
    
    # Processing metadata
    processing_time: float = Field(..., description="Total processing time in seconds")
    agents_used: list[str] = Field(..., description="List of agents that contributed")
    sources_consulted: list[str] = Field(default_factory=list, description="Policy documents consulted")
    
    # Case metadata
    case_number: Optional[str] = Field(None, description="OSB case identifier")
    case_priority: str = Field(default="medium", description="Case priority level")


class OSBErrorMessage(BaseModel):
    """Error message for OSB processing failures."""
    
    type: str = Field(default="osb_error", description="Message type")
    session_id: str = Field(..., description="Session identifier")
    error_type: str = Field(..., description="Type of error that occurred")
    error_message: str = Field(..., description="Human-readable error message")
    failed_agent: Optional[str] = Field(None, description="Agent that failed (if applicable)")
    recovery_options: list[Dict[str, Any]] = Field(default_factory=list, description="Available recovery options")
    retry_available: bool = Field(default=True, description="Whether retry is possible")


class OSBMessageProcessor:
    """
    Enhanced message processor for OSB-specific message types.
    
    Integrates with existing MessageService to handle OSB messages
    while maintaining compatibility with standard message processing.
    """
    
    @staticmethod
    def validate_osb_message(data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate OSB message structure and content.
        
        Args:
            data: Raw message data from WebSocket
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            message_type = data.get("type")
            
            if message_type == "run_flow" and data.get("flow") == "osb":
                # Validate OSB query message
                OSBQueryMessage(**data)
                return True, ""
            
            elif message_type in ["osb_status", "osb_partial", "osb_complete", "osb_error"]:
                # OSB-specific message types are valid
                return True, ""
            
            else:
                # Not an OSB message, but still valid
                return True, ""
                
        except Exception as e:
            return False, f"Invalid OSB message structure: {str(e)}"
    
    @staticmethod
    def enhance_run_request_for_osb(run_request: RunRequest, original_data: Dict[str, Any]) -> RunRequest:
        """
        Enhance RunRequest with OSB-specific parameters.
        
        Args:
            run_request: Original RunRequest
            original_data: Original WebSocket message data
            
        Returns:
            Enhanced RunRequest with OSB context
        """
        if run_request.flow != "osb":
            return run_request
        
        try:
            # Map 'criteria' to 'query' for OSB compatibility
            if 'criteria' in run_request.parameters and 'query' not in original_data:
                original_data['query'] = run_request.parameters.get('criteria', '')
                logger.debug(f"Mapped 'criteria' to 'query' for OSB flow: {original_data['query'][:100]}")
            
            # Parse OSB-specific fields from original data
            osb_message = OSBQueryMessage(**original_data)
            
            # Add OSB-specific parameters to the RunRequest
            osb_parameters = {
                "case_number": osb_message.case_number,
                "case_priority": osb_message.case_priority,
                "content_type": osb_message.content_type,
                "platform": osb_message.platform,
                "user_context": osb_message.user_context,
                "enable_multi_agent_synthesis": osb_message.enable_multi_agent_synthesis,
                "enable_cross_validation": osb_message.enable_cross_validation,
                "enable_precedent_analysis": osb_message.enable_precedent_analysis,
                "include_policy_references": osb_message.include_policy_references,
                "enable_streaming_response": osb_message.enable_streaming_response,
                "max_processing_time": osb_message.max_processing_time,
                "osb_query": osb_message.query,
                # Also keep 'query' for compatibility with agents that might use it
                "query": osb_message.query
            }
            
            # Merge with existing parameters
            enhanced_parameters = {**run_request.parameters, **osb_parameters}
            
            # Create enhanced RunRequest
            enhanced_request = run_request.model_copy(update={"parameters": enhanced_parameters})
            
            logger.info(f"Enhanced RunRequest for OSB flow with query: {osb_message.query[:100]}")
            return enhanced_request
            
        except Exception as e:
            logger.error(f"Error enhancing RunRequest for OSB: {e}")
            # If enhancement fails, at least map criteria to query
            if 'criteria' in run_request.parameters:
                run_request.parameters['query'] = run_request.parameters.get('criteria', '')
                run_request.parameters['osb_query'] = run_request.parameters.get('criteria', '')
                logger.info(f"Fallback: Mapped criteria to query for OSB flow")
            return run_request
    
    @staticmethod
    def create_osb_status_message(session_id: str, status: str, **kwargs) -> Dict[str, Any]:
        """
        Create OSB status update message.
        
        Args:
            session_id: Session identifier
            status: Current processing status
            **kwargs: Additional status parameters
            
        Returns:
            Dictionary representation of OSB status message
        """
        status_message = OSBStatusMessage(
            session_id=session_id,
            status=status,
            **kwargs
        )
        return status_message.dict()
    
    @staticmethod
    def create_osb_partial_response(session_id: str, agent: str, partial_response: str, 
                                  **kwargs) -> Dict[str, Any]:
        """
        Create OSB partial response message for streaming.
        
        Args:
            session_id: Session identifier
            agent: Agent providing the partial response
            partial_response: Partial response content
            **kwargs: Additional response parameters
            
        Returns:
            Dictionary representation of OSB partial response message
        """
        partial_message = OSBPartialResponseMessage(
            session_id=session_id,
            agent=agent,
            partial_response=partial_response,
            **kwargs
        )
        return partial_message.dict()
    
    @staticmethod
    def create_osb_complete_response(session_id: str, synthesis_summary: str, 
                                   agent_responses: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create OSB complete response message.
        
        Args:
            session_id: Session identifier
            synthesis_summary: Overall analysis synthesis
            agent_responses: Individual agent responses
            **kwargs: Additional response parameters
            
        Returns:
            Dictionary representation of OSB complete response message
        """
        complete_message = OSBCompleteResponseMessage(
            session_id=session_id,
            synthesis_summary=synthesis_summary,
            agent_responses=agent_responses,
            **kwargs
        )
        return complete_message.dict()
    
    @staticmethod
    def create_osb_error_message(session_id: str, error: Exception, **kwargs) -> Dict[str, Any]:
        """
        Create OSB error message with recovery options.
        
        Args:
            session_id: Session identifier
            error: Exception that occurred
            **kwargs: Additional error context
            
        Returns:
            Dictionary representation of OSB error message
        """
        error_message = OSBErrorMessage(
            session_id=session_id,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )
        return error_message.dict()


class OSBWebSocketStreamer:
    """
    Streaming support for OSB responses over WebSocket.
    
    Provides real-time updates during multi-agent OSB processing
    to keep users informed of progress and partial results.
    """
    
    def __init__(self, session: FlowRunContext):
        self.session = session
        self.websocket = session.websocket
        self.session_id = session.session_id
    
    async def send_status_update(self, status: str, **kwargs) -> None:
        """Send OSB status update to WebSocket client."""
        if not self.websocket:
            return
        
        try:
            status_message = OSBMessageProcessor.create_osb_status_message(
                self.session_id, status, **kwargs
            )
            await self.websocket.send_json(status_message)
            logger.debug(f"Sent OSB status update: {status} for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error sending OSB status update: {e}")
    
    async def send_partial_response(self, agent: str, partial_response: str, **kwargs) -> None:
        """Send OSB partial response to WebSocket client."""
        if not self.websocket:
            return
        
        try:
            partial_message = OSBMessageProcessor.create_osb_partial_response(
                self.session_id, agent, partial_response, **kwargs
            )
            await self.websocket.send_json(partial_message)
            logger.debug(f"Sent OSB partial response from {agent} for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error sending OSB partial response: {e}")
    
    async def send_complete_response(self, synthesis_summary: str, 
                                   agent_responses: Dict[str, Any], **kwargs) -> None:
        """Send OSB complete response to WebSocket client."""
        if not self.websocket:
            return
        
        try:
            complete_message = OSBMessageProcessor.create_osb_complete_response(
                self.session_id, synthesis_summary, agent_responses, **kwargs
            )
            await self.websocket.send_json(complete_message)
            logger.info(f"Sent OSB complete response for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error sending OSB complete response: {e}")
    
    async def send_error_message(self, error: Exception, **kwargs) -> None:
        """Send OSB error message to WebSocket client."""
        if not self.websocket:
            return
        
        try:
            error_message = OSBMessageProcessor.create_osb_error_message(
                self.session_id, error, **kwargs
            )
            await self.websocket.send_json(error_message)
            logger.error(f"Sent OSB error message for session {self.session_id}: {error}")
        except Exception as e:
            logger.error(f"Error sending OSB error message: {e}")


# Integration functions for existing message processing pipeline

def enhance_message_processing_for_osb(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Enhance existing message processing with OSB validation.
    
    This function can be called from MessageService.process_message_from_ui()
    to add OSB-specific validation before standard processing.
    
    Args:
        data: Raw WebSocket message data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    return OSBMessageProcessor.validate_osb_message(data)


def enhance_run_request_if_osb(run_request: RunRequest, original_data: Dict[str, Any]) -> RunRequest:
    """
    Enhance RunRequest with OSB parameters if it's an OSB flow.
    
    This function can be called from MessageService.process_message_from_ui()
    after creating the RunRequest to add OSB-specific context.
    
    Args:
        run_request: Original RunRequest
        original_data: Original WebSocket message data
        
    Returns:
        Enhanced RunRequest (or original if not OSB)
    """
    return OSBMessageProcessor.enhance_run_request_for_osb(run_request, original_data)