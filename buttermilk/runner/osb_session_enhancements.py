"""
OSB Session Enhancements

Enhanced session management functionality specifically for OSB (Oversight Board) 
interactive flows. Integrates with existing FlowRunner infrastructure to provide:

- OSB-specific session configuration
- Enhanced metadata tracking for OSB cases
- Performance monitoring for multi-agent OSB workflows
- Graceful error handling and recovery for OSB queries

This module extends the existing session management without replacing it,
following the principle of integration over creation.
"""

import asyncio
import time
from datetime import datetime, UTC
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from buttermilk._core.log import logger
from buttermilk._core.types import RunRequest
from buttermilk.runner.flowrunner import FlowRunContext, SessionStatus, SessionResources


@dataclass
class OSBSessionMetrics:
    """Metrics tracking for OSB session performance and usage."""
    
    query_count: int = 0
    total_processing_time: float = 0.0
    agent_response_times: Dict[str, list[float]] = field(default_factory=dict)
    error_count: int = 0
    cache_hit_rate: float = 0.0
    last_query_timestamp: Optional[datetime] = None
    
    def record_query_start(self) -> float:
        """Record the start of an OSB query and return timestamp."""
        self.query_count += 1
        self.last_query_timestamp = datetime.now(UTC)
        return time.time()
    
    def record_query_complete(self, start_time: float, agent_times: Dict[str, float] = None) -> None:
        """Record the completion of an OSB query."""
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        if agent_times:
            for agent_name, response_time in agent_times.items():
                if agent_name not in self.agent_response_times:
                    self.agent_response_times[agent_name] = []
                self.agent_response_times[agent_name].append(response_time)
    
    def record_error(self) -> None:
        """Record an error in OSB processing."""
        self.error_count += 1
    
    def get_average_processing_time(self) -> float:
        """Get average processing time per query."""
        return self.total_processing_time / max(self.query_count, 1)
    
    def get_agent_average_times(self) -> Dict[str, float]:
        """Get average response times per agent."""
        return {
            agent: sum(times) / len(times) 
            for agent, times in self.agent_response_times.items() 
            if times
        }


@dataclass 
class OSBCaseMetadata:
    """Metadata for OSB case tracking and audit trail."""
    
    case_number: Optional[str] = None
    case_priority: str = "medium"
    content_type: Optional[str] = None
    platform: Optional[str] = None
    user_context: Optional[str] = None
    policy_categories: list[str] = field(default_factory=list)
    precedent_cases: list[str] = field(default_factory=list)
    reviewer_notes: list[str] = field(default_factory=list)
    
    def add_reviewer_note(self, note: str, reviewer: str = "system") -> None:
        """Add a note to the case metadata."""
        timestamp = datetime.now(UTC).isoformat()
        formatted_note = f"[{timestamp}] {reviewer}: {note}"
        self.reviewer_notes.append(formatted_note)


class OSBSessionEnhancer:
    """
    Enhances existing FlowRunContext sessions with OSB-specific functionality.
    
    This class works with the existing session management infrastructure
    and adds OSB-specific features without replacing core functionality.
    """
    
    @staticmethod
    def enhance_session_for_osb(session: FlowRunContext, osb_config: Dict[str, Any]) -> None:
        """
        Enhance an existing session with OSB-specific capabilities.
        
        Args:
            session: Existing FlowRunContext to enhance
            osb_config: OSB configuration from osb.yaml
        """
        # Add OSB-specific attributes to the session's custom resources
        session.add_custom_resource("osb_metrics", OSBSessionMetrics())
        session.add_custom_resource("osb_case_metadata", OSBCaseMetadata())
        session.add_custom_resource("osb_config", osb_config)
        
        # Configure session timeout based on OSB configuration
        if "session_management" in osb_config.get("parameters", {}):
            session_mgmt = osb_config["parameters"]["session_management"]
            session.session_timeout = session_mgmt.get("session_timeout", 3600)
        
        logger.info(f"Enhanced session {session.session_id} for OSB flow with timeout {session.session_timeout}s")
    
    @staticmethod
    def process_osb_run_request(run_request: RunRequest, session: FlowRunContext) -> RunRequest:
        """
        Process and enhance a RunRequest for OSB flow.
        
        Args:
            run_request: Original RunRequest
            session: Enhanced session context
            
        Returns:
            Enhanced RunRequest with OSB-specific parameters
        """
        if run_request.flow != "osb":
            return run_request
        
        # Extract OSB-specific parameters from the request
        osb_metadata = session.resources.custom_resources.get("osb_case_metadata")
        osb_config = session.resources.custom_resources.get("osb_config", {})
        
        # Enhanced RunRequest with OSB context
        enhanced_request = run_request.model_copy(deep=True)
        
        # Add OSB session context to the request
        if not hasattr(enhanced_request, 'osb_context'):
            enhanced_request.osb_context = {}
        
        enhanced_request.osb_context = {
            "session_id": session.session_id,
            "case_metadata": osb_metadata.__dict__ if osb_metadata else {},
            "osb_features": osb_config.get("parameters", {}).get("osb_features", {}),
            "performance_config": osb_config.get("parameters", {}).get("performance", {}),
            "error_handling": osb_config.get("parameters", {}).get("error_handling", {})
        }
        
        logger.debug(f"Enhanced RunRequest for OSB flow in session {session.session_id}")
        return enhanced_request
    
    @staticmethod
    def record_osb_query_metrics(session: FlowRunContext, query_start_time: float, 
                                agent_responses: Dict[str, Any] = None) -> None:
        """
        Record performance metrics for an OSB query.
        
        Args:
            session: Session context with OSB enhancements
            query_start_time: Timestamp when query processing started
            agent_responses: Dictionary of agent responses with timing data
        """
        osb_metrics = session.resources.custom_resources.get("osb_metrics")
        if not osb_metrics:
            logger.warning(f"No OSB metrics found for session {session.session_id}")
            return
        
        # Extract agent timing data if available
        agent_times = {}
        if agent_responses:
            for agent_name, response in agent_responses.items():
                if isinstance(response, dict) and "processing_time" in response:
                    agent_times[agent_name] = response["processing_time"]
        
        osb_metrics.record_query_complete(query_start_time, agent_times)
        
        # Log performance summary
        avg_time = osb_metrics.get_average_processing_time()
        agent_avgs = osb_metrics.get_agent_average_times()
        
        logger.info(f"OSB Query Complete - Session: {session.session_id}, "
                   f"Processing Time: {time.time() - query_start_time:.2f}s, "
                   f"Session Avg: {avg_time:.2f}s, "
                   f"Agent Avgs: {agent_avgs}")
    
    @staticmethod
    def handle_osb_error(session: FlowRunContext, error: Exception, 
                        error_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle errors in OSB processing with graceful degradation.
        
        Args:
            session: Session context with OSB enhancements
            error: The exception that occurred
            error_context: Additional context about the error
            
        Returns:
            Error response with recovery information
        """
        osb_metrics = session.resources.custom_resources.get("osb_metrics")
        osb_config = session.resources.custom_resources.get("osb_config", {})
        
        if osb_metrics:
            osb_metrics.record_error()
        
        error_handling = osb_config.get("parameters", {}).get("error_handling", {})
        enable_graceful_degradation = error_handling.get("enable_graceful_degradation", True)
        
        error_response = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "session_id": session.session_id,
            "graceful_degradation": enable_graceful_degradation,
            "recovery_options": []
        }
        
        # Add recovery options based on error type and configuration
        if enable_graceful_degradation:
            if "agent" in error_context:
                failed_agent = error_context["agent"]
                available_agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]
                remaining_agents = [a for a in available_agents if a != failed_agent]
                
                error_response["recovery_options"].append({
                    "type": "continue_without_agent",
                    "failed_agent": failed_agent,
                    "available_agents": remaining_agents
                })
            
            if "vector_store" in str(error).lower():
                error_response["recovery_options"].append({
                    "type": "retry_with_backoff",
                    "max_retries": error_handling.get("max_retries", 3),
                    "backoff_factor": error_handling.get("retry_backoff_factor", 2)
                })
        
        logger.error(f"OSB Error in session {session.session_id}: {error}", 
                    extra={"error_context": error_context, "recovery_options": error_response["recovery_options"]})
        
        return error_response
    
    @staticmethod
    def get_osb_session_status(session: FlowRunContext) -> Dict[str, Any]:
        """
        Get comprehensive status information for an OSB session.
        
        Args:
            session: Session context with OSB enhancements
            
        Returns:
            Dictionary with detailed session status
        """
        osb_metrics = session.resources.custom_resources.get("osb_metrics")
        osb_case_metadata = session.resources.custom_resources.get("osb_case_metadata")
        osb_config = session.resources.custom_resources.get("osb_config", {})
        
        status = {
            "session_id": session.session_id,
            "flow_name": session.flow_name,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "is_expired": session.is_expired()
        }
        
        # Add OSB-specific metrics if available
        if osb_metrics:
            status["metrics"] = {
                "query_count": osb_metrics.query_count,
                "average_processing_time": osb_metrics.get_average_processing_time(),
                "error_count": osb_metrics.error_count,
                "agent_performance": osb_metrics.get_agent_average_times(),
                "last_query": osb_metrics.last_query_timestamp.isoformat() if osb_metrics.last_query_timestamp else None
            }
        
        # Add case metadata if available
        if osb_case_metadata:
            status["case_metadata"] = {
                "case_number": osb_case_metadata.case_number,
                "case_priority": osb_case_metadata.case_priority,
                "content_type": osb_case_metadata.content_type,
                "policy_categories": osb_case_metadata.policy_categories,
                "reviewer_notes_count": len(osb_case_metadata.reviewer_notes)
            }
        
        # Add configuration status
        if osb_config:
            config_status = osb_config.get("parameters", {})
            status["osb_features"] = {
                "multi_agent_synthesis": config_status.get("enable_multi_agent_synthesis", False),
                "cross_validation": config_status.get("enable_cross_validation", False),
                "case_tracking": config_status.get("osb_features", {}).get("enable_case_tracking", False),
                "policy_references": config_status.get("osb_features", {}).get("enable_policy_references", False)
            }
        
        return status


# Integration hooks for existing FlowRunner
class OSBFlowIntegration:
    """
    Integration hooks to seamlessly add OSB functionality to existing FlowRunner.
    
    These methods can be called from the existing FlowRunner methods to add
    OSB-specific behavior without modifying the core infrastructure.
    """
    
    @staticmethod
    async def enhance_session_if_osb(session: FlowRunContext, run_request: RunRequest, 
                                   flow_configs: Dict[str, Any]) -> None:
        """
        Enhance session with OSB functionality if it's an OSB flow.
        
        This method should be called from FlowRunner.run_flow() after session creation.
        """
        if run_request.flow == "osb" and "osb" in flow_configs:
            OSBSessionEnhancer.enhance_session_for_osb(session, flow_configs["osb"])
            
            # Record query start metrics
            osb_metrics = session.resources.custom_resources.get("osb_metrics")
            if osb_metrics:
                query_start_time = osb_metrics.record_query_start()
                session.add_custom_resource("osb_query_start_time", query_start_time)
    
    @staticmethod
    async def process_osb_request(run_request: RunRequest, session: FlowRunContext) -> RunRequest:
        """
        Process RunRequest for OSB-specific enhancements.
        
        This method should be called from FlowRunner.run_flow() before orchestrator creation.
        """
        if run_request.flow == "osb":
            return OSBSessionEnhancer.process_osb_run_request(run_request, session)
        return run_request
    
    @staticmethod
    async def handle_osb_completion(session: FlowRunContext, orchestrator_result: Any) -> None:
        """
        Handle OSB flow completion with metrics and cleanup.
        
        This method should be called from FlowRunner.run_flow() after flow completion.
        """
        if session.flow_name == "osb":
            query_start_time = session.resources.custom_resources.get("osb_query_start_time")
            if query_start_time:
                # Extract agent response data from orchestrator result if available
                agent_responses = None
                if hasattr(orchestrator_result, 'agent_responses'):
                    agent_responses = orchestrator_result.agent_responses
                
                OSBSessionEnhancer.record_osb_query_metrics(session, query_start_time, agent_responses)
    
    @staticmethod
    async def handle_osb_error(session: FlowRunContext, error: Exception, 
                             error_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle OSB flow errors with graceful degradation.
        
        This method should be called from FlowRunner error handlers.
        """
        if session.flow_name == "osb":
            return OSBSessionEnhancer.handle_osb_error(session, error, error_context)
        return {"status": "error", "message": str(error)}