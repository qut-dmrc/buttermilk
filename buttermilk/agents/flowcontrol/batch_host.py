"""
Defines a sequential batch host that executes a predefined sequence of steps without human interaction.
"""

import asyncio
from typing import Optional, Dict, Any, List, Union, cast, AsyncGenerator

from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.contract import (
    END,
    WAIT,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    StepRequest,
)
from buttermilk.agents.flowcontrol.host import LLMHostAgent


class SequentialBatchHost(LLMHostAgent):
    """
    A host agent for batch processing with minimal human interaction.
    
    This agent is designed for scenarios where a fixed sequence of steps needs to
    be executed without user input at each step. It's ideal for batch processing,
    automated pipelines, and scheduled tasks. The host follows a predefined
    sequence of steps, executes them directly, and only involves a human when
    explicitly configured to do so.
    """
    
    # Configuration for sequential processing
    sequence: List[str] = Field(
        default_factory=list,
        description="Predefined sequence of agent roles to execute in order"
    )
    wait_between_steps: float = Field(
        default=0.5,
        description="Time in seconds to wait between steps"
    )
    max_steps: int = Field(
        default=100,
        description="Maximum number of steps to execute before automatically ending"
    )
    continue_on_error: bool = Field(
        default=False,
        description="Whether to continue execution if a step fails"
    )
    
    # Override human_in_loop to default to False
    human_in_loop: bool = Field(
        default=False,
        description="Whether to interact with the human/manager for step confirmation"
    )
    
    # Additional state
    _current_sequence_index: int = PrivateAttr(default=0)
    _step_count: int = PrivateAttr(default=0)
    _error_count: int = PrivateAttr(default=0)
    
    async def initialize(self, input_callback=None, **kwargs) -> None:
        """Initialize the agent with a specific step sequence if provided"""
        self._current_sequence_index = 0
        self._step_count = 0
        self._error_count = 0
        await super().initialize(input_callback=input_callback, **kwargs)
    
    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """
        Generate steps from the predefined sequence.
        
        This implementation processes steps in the order specified in the sequence
        configuration. If no sequence is provided, it will be initialized from the
        participants when available.
        """
        # Wait for participants to be available if sequence is empty
        if not self.sequence:
            while self._participants is None:
                await asyncio.sleep(0.1)
            
            # Initialize from participants
            self.sequence = list(self._participants.keys())
            logger.info(f"Initialized sequence from participants: {self.sequence}")
            
        # Generate steps from the sequence
        step_num = 0
        while step_num < self.max_steps:
            step_num += 1
            
            # Cycle through the sequence
            if self._current_sequence_index >= len(self.sequence):
                self._current_sequence_index = 0
                
                # End if we've gone through the whole sequence
                if step_num > len(self.sequence):
                    logger.info(f"Completed full sequence of {len(self.sequence)} steps")
                    yield StepRequest(role=END, content="Completed all steps in sequence")
                    break
            
            # Get the next role in sequence
            role = self.sequence[self._current_sequence_index]
            self._current_sequence_index += 1
            
            # Generate the step
            yield StepRequest(
                role=role, 
                content=f"Batch step {step_num} calling {role} (sequence position {self._current_sequence_index})"
            )
            
        # If max_steps is reached
        if step_num >= self.max_steps:
            logger.info(f"Reached maximum number of steps ({self.max_steps})")
            yield StepRequest(role=END, content=f"Reached maximum steps limit of {self.max_steps}")
    
    async def _should_execute_directly(self, step: StepRequest) -> bool:
        """Always execute steps directly in batch mode"""
        return True
    
    async def _execute_step(self, step: StepRequest) -> None:
        """
        Execute a step with direct message sending to the target agent.
        
        This method handles publishing the message to the appropriate topic
        and tracking completion status.
        """
        if step.role == END or step.role == WAIT:
            logger.debug(f"Skipping execution for control step: {step.role}")
            return
            
        logger.info(f"Batch executing step for role: {step.role}")
        self._step_count += 1
        
        try:
            # Create message for the agent
            message = AgentInput(prompt=step.prompt, records=getattr(self, '_records', []))
            
            # TODO: Implement direct message publishing
            # This is a placeholder for the actual implementation
            # which would directly publish to the role's topic or send to agents
            
            # Update tracking state
            self._current_step_name = step.role
            self._expected_agents_current_step.clear()
            self._completed_agents_current_step.clear()
            
            # Pause briefly between steps
            await asyncio.sleep(self.wait_between_steps)
            
        except Exception as e:
            logger.error(f"Error executing batch step for {step.role}: {e}")
            self._error_count += 1
            if not self.continue_on_error:
                raise
    
    def _build_conductor_context(self, message: ConductorRequest) -> Dict[str, Any]:
        """Add batch processing metrics to the context"""
        context = super()._build_conductor_context(message)
        
        # Add batch-specific stats
        context.update({
            "batch_stats": {
                "steps_executed": self._step_count,
                "errors": self._error_count,
                "sequence_position": self._current_sequence_index,
                "sequence": self.sequence,
            }
        })
        
        return context
