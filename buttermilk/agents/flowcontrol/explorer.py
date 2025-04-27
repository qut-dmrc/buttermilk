import asyncio
from typing import Optional, Dict, Any, List, Union, cast, AsyncGenerator

from pydantic import BaseModel, Field

from autogen_core import CancellationToken
from buttermilk import logger
from buttermilk._core.contract import (
    END,
    WAIT,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    ErrorEvent,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
    ToolOutput,
)
from .llmhost import LLMHostAgent

TRUNCATE_LEN = 1000  # characters per history message


class ExplorerHost(LLMHostAgent):
    """
    An advanced host agent for interactive exploration workflows.
    
    This agent specializes in guiding conversations where the path isn't predetermined,
    but rather discovered through interaction and exploration. It maintains state
    about the exploration history, understands user feedback, and adapts its
    recommendations based on the evolving context.
    """
    
    _output_model: Optional[type[BaseModel]] = StepRequest
    
    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent:
        """Process a message for the Explorer agent - calls _choose to determine the next step"""
        if isinstance(message, ConductorRequest):
            return await self._choose(message=message)
        return StepRequest(role=WAIT, content="Waiting for conductor request")
    
    # Explorer-specific configuration
    exploration_mode: str = Field(
        default="interactive",
        description="Mode of exploration: 'interactive' (with user feedback) or 'autonomous'",
    )
    max_exploration_steps: int = Field(
        default=20, 
        description="Maximum number of exploration steps before suggesting completion"
    )
    consider_previous_steps: bool = Field(
        default=True,
        description="Whether to consider previous steps when choosing the next one"
    )
    prioritize_unexplored: bool = Field(
        default=True,
        description="Whether to prioritize unexplored agents over previously used ones"
    )

    async def _choose(self, message: ConductorRequest) -> StepRequest:
        """
        Choose the next step based on exploration context.
        
        This method analyzes the conversation history, user feedback,
        and exploration state to determine the most appropriate next step.
        It uses LLM reasoning through the _process method, which employs
        the agent's language model to make decisions.
        
        Args:
            message: The ConductorRequest containing context for decision
            
        Returns:
            A StepRequest representing the chosen next step
        """
        # Check if we've reached the maximum exploration steps
        if len(self._exploration_path) >= self.max_exploration_steps:
            logger.info(f"Reached maximum exploration steps ({self.max_exploration_steps}), suggesting END")
            return StepRequest(role=END, content="Reached maximum exploration steps")
        
        # Enhance message context with exploration-specific information
        enhanced_message = await self._enhance_message_for_exploration(message)
        
        # Use LLM to determine the next step
        result = await self._process(message=enhanced_message)
        
        # Process the result - since we've updated our type system, result should now be directly a StepRequest
        # But keep the fallback logic for backward compatibility
        if isinstance(result, StepRequest):
            step = result
        elif isinstance(result, AgentOutput) and hasattr(result, 'outputs') and isinstance(result.outputs, StepRequest):
            # Legacy code path - the LLM returned a StepRequest wrapped in AgentOutput
            # In the future, this branch can be removed as agents are updated
            step = result.outputs
        else:
            # Fallback for invalid or unexpected return types
            logger.warning(f"Explorer received unexpected result type from LLM: {type(result)}")
            step = StepRequest(role=WAIT, content="Waiting after receiving invalid result type")
        
        # Validate the step has a role
        if not step.role:
            logger.warning("Explorer received step without role from LLM, using fallback")
            step = StepRequest(role=WAIT, content="Waiting after receiving invalid step suggestion")
            
        return step
    
    async def _enhance_message_for_exploration(self, message: ConductorRequest) -> ConductorRequest:
        """
        Enhance the conductor request with exploration-specific context.
        
        This method adds information about exploration history, which roles have been
        explored, user feedback patterns, and suggested areas for further exploration.
        
        Args:
            message: The original ConductorRequest
            
        Returns:
            An enhanced ConductorRequest with exploration context
        """
        # Create a copy to avoid modifying the original
        enhanced = message.model_copy(deep=True)
        
        # Get all roles that have been explored so far
        explored_roles = set()
        for path_id in self._exploration_path:
            if path_id in self._exploration_results:
                role = self._exploration_results[path_id].get("role")
                if role:
                    explored_roles.add(role)
        
        # Get all available roles from participants
        available_roles = set(self._participants.keys())
        
        # Find unexplored roles
        unexplored_roles = available_roles - explored_roles
        
        # Add exploration analytics to the inputs
        exploration_context = {
            "exploration_statistics": {
                "steps_taken": len(self._exploration_path),
                "max_steps": self.max_exploration_steps,
                "explored_roles": list(explored_roles),
                "unexplored_roles": list(unexplored_roles),
                "available_roles": list(available_roles),
            },
            "recent_user_feedback": self._user_feedback,
            "exploration_history": self._summarize_exploration_history()
        }
        
        # Update inputs with exploration context
        if "inputs" in enhanced.__dict__:
            if isinstance(enhanced.inputs, dict):
                enhanced.inputs.update(exploration_context)
            else:
                # If inputs is not a dict, create a new one
                enhanced.inputs = {**exploration_context}
        
        # Update the prompt to encourage exploration if needed
        if self.prioritize_unexplored and unexplored_roles:
            prioritize_message = f"Consider exploring these roles that haven't been used yet: {', '.join(unexplored_roles)}. "
            enhanced.prompt = prioritize_message + (enhanced.prompt or "")
            
        return enhanced
    
    def _summarize_exploration_history(self) -> List[Dict[str, Any]]:
        """
        Create a summarized version of the exploration history.
        
        This creates a chronological summary of exploration steps taken,
        focusing on role sequence, outcomes, and patterns.
        
        Returns:
            A list of summary dictionaries for each exploration step
        """
        summary = []
        for step_id in self._exploration_path:
            if step_id in self._exploration_results:
                result = self._exploration_results[step_id]
                step_summary = {
                    "id": step_id,
                    "role": result.get("role", "unknown"),
                    "success": not result.get("is_error", False),
                }
                
                # Add a condensed output summary if available
                if "outputs" in result:
                    outputs = result["outputs"]
                    if isinstance(outputs, str):
                        # Truncate long string outputs
                        step_summary["output_summary"] = outputs[:100] + "..." if len(outputs) > 100 else outputs
                    elif isinstance(outputs, (dict, list)):
                        # For structured data, just note the type
                        step_summary["output_summary"] = f"{type(outputs).__name__} data"
                    
                summary.append(step_summary)
                
        return summary

