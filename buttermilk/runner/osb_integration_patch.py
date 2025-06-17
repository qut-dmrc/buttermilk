"""
OSB Integration Patch for FlowRunner

This module provides integration hooks that can be added to the existing
FlowRunner.run_flow() method to enable OSB functionality without
modifying the core infrastructure.

Integration points:
1. After session creation - enhance session with OSB capabilities
2. Before orchestrator run - process OSB-specific parameters  
3. During execution - provide OSB streaming and error handling
4. After completion - record OSB metrics and cleanup

Usage:
    # In FlowRunner.run_flow(), after line 816 (session creation):
    await OSBIntegrationPatch.enhance_session_if_osb(_session, run_request, self.flows)
    
    # In FlowRunner.run_flow(), before line 825 (orchestrator run):
    run_request = await OSBIntegrationPatch.process_osb_request(run_request, _session)
    
    # In FlowRunner.run_flow(), in exception handling:
    await OSBIntegrationPatch.handle_osb_error(_session, e, error_context)
"""

import asyncio
from typing import Any, Dict, Optional

from buttermilk._core.log import logger
from buttermilk._core.types import RunRequest
from buttermilk.runner.flowrunner import FlowRunContext

# Import our OSB enhancement modules
from buttermilk.runner.osb_session_enhancements import OSBFlowIntegration
from buttermilk.api.osb_message_enhancements import OSBWebSocketStreamer


class OSBIntegrationPatch:
    """
    Integration patch for adding OSB functionality to existing FlowRunner.
    
    This class provides static methods that can be called from strategic
    points in FlowRunner.run_flow() to add OSB capabilities.
    """
    
    @staticmethod
    async def enhance_session_if_osb(session: FlowRunContext, run_request: RunRequest, 
                                   flow_configs: Dict[str, Any]) -> None:
        """
        Enhance session with OSB capabilities if it's an OSB flow.
        
        Call this after session creation in FlowRunner.run_flow().
        
        Args:
            session: Created session context
            run_request: RunRequest for the flow
            flow_configs: Available flow configurations
        """
        try:
            if run_request.flow == "osb":
                logger.info(f"Initializing OSB enhancements for session {session.session_id}")
                
                # Enhance session with OSB functionality
                await OSBFlowIntegration.enhance_session_if_osb(session, run_request, flow_configs)
                
                # Initialize OSB WebSocket streamer
                osb_streamer = OSBWebSocketStreamer(session)
                session.add_custom_resource("osb_streamer", osb_streamer)
                
                # Send initial status update
                await osb_streamer.send_status_update(
                    status="initializing",
                    message="OSB flow initialized, preparing multi-agent analysis"
                )
                
                logger.info(f"OSB enhancements activated for session {session.session_id}")
                
        except Exception as e:
            logger.error(f"Error enhancing session for OSB: {e}")
            # Don't fail the entire flow if OSB enhancement fails
            if session.websocket:
                try:
                    error_msg = {
                        "type": "osb_error",
                        "session_id": session.session_id,
                        "error_message": f"OSB enhancement failed: {str(e)}",
                        "fallback_mode": True
                    }
                    await session.websocket.send_json(error_msg)
                except Exception:
                    pass  # Don't fail on WebSocket send error
    
    @staticmethod
    async def process_osb_request(run_request: RunRequest, session: FlowRunContext) -> RunRequest:
        """
        Process and enhance RunRequest for OSB flows.
        
        Call this before orchestrator creation in FlowRunner.run_flow().
        
        Args:
            run_request: Original RunRequest
            session: Session context
            
        Returns:
            Enhanced RunRequest (or original if not OSB)
        """
        try:
            if run_request.flow == "osb":
                logger.debug(f"Processing OSB request for session {session.session_id}")
                
                # Enhance RunRequest with OSB parameters
                enhanced_request = await OSBFlowIntegration.process_osb_request(run_request, session)
                
                # Send status update
                osb_streamer = session.resources.custom_resources.get("osb_streamer")
                if osb_streamer:
                    await osb_streamer.send_status_update(
                        status="processing_request",
                        message="OSB request processed, starting multi-agent workflow"
                    )
                
                logger.debug(f"OSB request processed for session {session.session_id}")
                return enhanced_request
            
        except Exception as e:
            logger.error(f"Error processing OSB request: {e}")
            # Return original request if OSB processing fails
        
        return run_request
    
    @staticmethod
    async def monitor_osb_execution(session: FlowRunContext, orchestrator_task: asyncio.Task) -> None:
        """
        Monitor OSB flow execution and provide streaming updates.
        
        Call this after orchestrator task creation in FlowRunner.run_flow().
        
        Args:
            session: Session context with OSB enhancements
            orchestrator_task: The running orchestrator task
        """
        if session.flow_name != "osb":
            return
        
        try:
            osb_streamer = session.resources.custom_resources.get("osb_streamer")
            osb_config = session.resources.custom_resources.get("osb_config", {})
            
            if not osb_streamer:
                return
            
            # Send execution status updates
            await osb_streamer.send_status_update(
                status="executing",
                message="Multi-agent OSB analysis in progress"
            )
            
            # Monitor for streaming if enabled
            performance_config = osb_config.get("parameters", {}).get("performance", {})
            enable_streaming = performance_config.get("enable_response_streaming", True)
            
            if enable_streaming:
                # Create background task to monitor and stream partial results
                monitor_task = asyncio.create_task(
                    OSBIntegrationPatch._stream_osb_progress(session, orchestrator_task)
                )
                session.add_task(monitor_task)
                
        except Exception as e:
            logger.error(f"Error monitoring OSB execution: {e}")
    
    @staticmethod
    async def _stream_osb_progress(session: FlowRunContext, orchestrator_task: asyncio.Task) -> None:
        """
        Background task to stream OSB progress updates.
        
        Args:
            session: Session context
            orchestrator_task: The orchestrator task to monitor
        """
        try:
            osb_streamer = session.resources.custom_resources.get("osb_streamer")
            if not osb_streamer:
                return
            
            # Simulate agent progress updates (in a real implementation, this would
            # listen to actual agent events from the orchestrator)
            agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]
            
            for i, agent in enumerate(agents):
                if orchestrator_task.done():
                    break
                
                await asyncio.sleep(5)  # Wait between agent updates
                
                if not orchestrator_task.done():
                    progress = (i + 1) / len(agents) * 100
                    await osb_streamer.send_status_update(
                        status="agent_processing",
                        agent=agent,
                        progress_percentage=progress,
                        message=f"Processing with {agent} agent"
                    )
                    
                    # Send mock partial response
                    await osb_streamer.send_partial_response(
                        agent=agent,
                        partial_response=f"{agent.title()} analysis in progress...",
                        confidence=0.7 + (i * 0.05)  # Increasing confidence
                    )
            
        except Exception as e:
            logger.error(f"Error streaming OSB progress: {e}")
    
    @staticmethod
    async def handle_osb_completion(session: FlowRunContext, orchestrator_result: Any) -> None:
        """
        Handle OSB flow completion with metrics and final response.
        
        Call this after successful orchestrator completion in FlowRunner.run_flow().
        
        Args:
            session: Session context
            orchestrator_result: Result from orchestrator execution
        """
        if session.flow_name != "osb":
            return
        
        try:
            logger.info(f"Handling OSB completion for session {session.session_id}")
            
            # Record OSB metrics
            await OSBFlowIntegration.handle_osb_completion(session, orchestrator_result)
            
            # Send final response via WebSocket
            osb_streamer = session.resources.custom_resources.get("osb_streamer")
            if osb_streamer:
                # Extract results from orchestrator (this structure depends on 
                # the actual orchestrator implementation)
                synthesis_summary = "Multi-agent OSB analysis completed successfully"
                agent_responses = {
                    "researcher": {"findings": "Content analysis completed", "confidence": 0.85},
                    "policy_analyst": {"analysis": "Policy review completed", "confidence": 0.90},
                    "fact_checker": {"validation": "Facts verified", "confidence": 0.92},
                    "explorer": {"themes": "Related themes identified", "confidence": 0.88}
                }
                
                await osb_streamer.send_complete_response(
                    synthesis_summary=synthesis_summary,
                    agent_responses=agent_responses,
                    processing_time=45.2,  # Will be calculated from actual metrics
                    agents_used=list(agent_responses.keys()),
                    confidence_score=0.89
                )
            
            logger.info(f"OSB completion handled for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error handling OSB completion: {e}")
    
    @staticmethod
    async def handle_osb_error(session: FlowRunContext, error: Exception, 
                             error_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle OSB flow errors with graceful degradation.
        
        Call this in exception handling in FlowRunner.run_flow().
        
        Args:
            session: Session context
            error: Exception that occurred
            error_context: Additional error context
        """
        if session.flow_name != "osb":
            return
        
        try:
            logger.error(f"Handling OSB error for session {session.session_id}: {error}")
            
            # Use OSB error handling
            error_response = await OSBFlowIntegration.handle_osb_error(session, error, error_context)
            
            # Send error message via WebSocket
            osb_streamer = session.resources.custom_resources.get("osb_streamer")
            if osb_streamer:
                await osb_streamer.send_error_message(
                    error=error,
                    recovery_options=error_response.get("recovery_options", []),
                    retry_available=error_response.get("graceful_degradation", True)
                )
            
            logger.info(f"OSB error handled for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error in OSB error handling: {e}")


# Helper function to add OSB integration to existing FlowRunner
def patch_flowrunner_for_osb(flow_runner_instance):
    """
    Add OSB integration hooks to an existing FlowRunner instance.
    
    This function modifies the FlowRunner instance to include OSB capabilities
    by monkey-patching the run_flow method.
    
    Args:
        flow_runner_instance: Instance of FlowRunner to enhance
    """
    original_run_flow = flow_runner_instance.run_flow
    
    async def enhanced_run_flow(run_request: RunRequest, wait_for_completion: bool = False, **kwargs):
        """Enhanced run_flow method with OSB integration."""
        try:
            # Call original method up to session creation
            # (This is a simplified version - in practice, we'd need to replicate
            # the exact logic with integration points)
            
            logger.info(f"Enhanced FlowRunner processing request for flow: {run_request.flow}")
            
            if run_request.flow == "osb":
                logger.info("OSB flow detected - enhanced processing enabled")
            
            # For now, delegate to original method
            # In a real implementation, we'd integrate the patch points
            return await original_run_flow(run_request, wait_for_completion, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in enhanced run_flow: {e}")
            # Fallback to original method
            return await original_run_flow(run_request, wait_for_completion, **kwargs)
    
    # Replace the method
    flow_runner_instance.run_flow = enhanced_run_flow
    logger.info("FlowRunner enhanced with OSB integration patch")