import asyncio
from typing import Optional, Dict, Any, List, Union, cast, AsyncGenerator

from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.contract import (
    END,
    WAIT,
    AgentOutput,
    ConductorRequest,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
)
from buttermilk.agents.flowcontrol.host import LLMHostAgent

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
            return StepRequest(role=END, description="Reached maximum exploration steps")
        
        # Enhance message context with exploration-specific information
        enhanced_message = await self._enhance_message_for_exploration(message)
        
        # Use LLM to determine the next step
        result = await self._process(message=enhanced_message)
        
        # Process the result to ensure we return a valid StepRequest
        if isinstance(result, StepRequest):
            step = result
        elif isinstance(result, AgentOutput) and hasattr(result, 'outputs') and isinstance(result.outputs, StepRequest):
            step = result.outputs
        else:
            # Fallback for invalid or unexpected return types
            logger.warning(f"Explorer received unexpected result type from LLM: {type(result)}")
            step = StepRequest(role=WAIT, description="Waiting after receiving invalid result type")
        
        # Validate the step has a role
        if not step.role:
            logger.warning("Explorer received step without role from LLM, using fallback")
            step = StepRequest(role=WAIT, description="Waiting after receiving invalid step suggestion")
            
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


class SequentialHost(LLMHostAgent):
    """
    A simple host agent that sequentially steps through a predefined list of agents.
    
    This host is designed for batch processing and pipeline scenarios where the
    flow is predetermined and minimal human interaction is desired. It follows
    a fixed sequence of steps defined either at initialization or extracted 
    from participant information.
    """
    
    _output_model: Optional[type[BaseModel]] = StepRequest
    
    # Sequential host configuration
    sequence: List[str] = Field(
        default_factory=list,
        description="Predefined sequence of agent roles to execute in order"
    )
    repeat_sequence: bool = Field(
        default=False,
        description="Whether to restart the sequence when completed"
    )
    skip_errors: bool = Field(
        default=True,
        description="Whether to continue execution if a step fails"
    )
    
    # Override human_in_loop to default to False for this host
    human_in_loop: bool = Field(
        default=False,
        description="Whether to interact with the human/manager for step confirmation"
    )
    
    async def initialize(self, input_callback=None, **kwargs) -> None:
        """Initialize the agent with a specific step sequence if provided"""
        self._current_sequence_index = 0
        await super().initialize(input_callback=input_callback, **kwargs)
        
    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """
        Generate steps from the predefined sequence.
        
        This override replaces the default generator approach with a simple
        index-based approach using the predefined sequence list.
        """
        # Use the predefined sequence if available
        if self.sequence:
            while True:
                if self._current_sequence_index >= len(self.sequence):
                    if self.repeat_sequence:
                        # Start over
                        self._current_sequence_index = 0
                    else:
                        # End the sequence
                        yield StepRequest(role=END, description="Sequence completed")
                        break
                
                # Get the next role in sequence
                role = self.sequence[self._current_sequence_index]
                self._current_sequence_index += 1
                
                yield StepRequest(role=role, description=f"Sequential step {self._current_sequence_index} calling {role}")
        else:
            # Fall back to the parent implementation if no sequence defined
            async for step in super()._sequence():
                yield step

    async def _choose(self, message: ConductorRequest) -> StepRequest:
        """
        Choose the next step based on the predefined sequence.
        
        This method simply returns the next step in the sequence without
        any complex decision making.
        
        Args:
            message: The ConductorRequest (mostly ignored in this implementation)
            
        Returns:
            A StepRequest for the next step in sequence
        """
        # If sequence is empty but we have participants, initialize from participants
        if not self.sequence and self._participants:
            self.sequence = list(self._participants.keys())
            self._current_sequence_index = 0
            logger.info(f"Initialized sequence from participants: {self.sequence}")
        
        # Get the next step from the sequence generator
        try:
            step = await anext(self._step_generator)
            return step
        except StopAsyncIteration:
            # End the sequence if the generator is exhausted
            return StepRequest(role=END, description="Sequence exhausted")
