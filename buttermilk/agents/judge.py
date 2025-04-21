import asyncio
import json
import pprint
from typing import Any, AsyncGenerator, Callable, Literal, Optional, Self, Dict, List, Union, Type

# Import necessary Autogen types for type hints, but not agent classes
from autogen_core import CancellationToken, MessageContext
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage
from pydantic import BaseModel, Field

# Buttermilk core imports
from buttermilk._core.agent import AgentInput, AgentOutput, buttermilk_handler  # Import decorator
from buttermilk._core.config import ToolConfig  # Import ToolConfig needed by LLMAgent's Pydantic model
from buttermilk._core.contract import LLMMessage, ToolOutput, AgentInput, AgentOutput  # Add AgentInput/Output here too
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import CreateResult
from buttermilk._core.types import Record
from buttermilk.agents.llm import LLMAgent
from buttermilk.bm import bm, logger

# Utilities
from buttermilk.utils.templating import _parse_prompty, load_template, make_messages


# --- Pydantic Models ---
class AgentRequest(BaseModel):
    """Defines the expected structure of the 'inputs' field for a Judge task."""

    prompt: str = Field(..., description="The core question or instruction for the Judge.")
    # Records are typically passed via the top-level AgentInput.records


class AgentReasons(BaseModel):
    """Defines the expected structure of the 'outputs' field for a Judge task."""

    conclusion: str = Field(..., description="Your conclusion or final answer.")
    prediction: bool = Field(
        description="True if the content violates the policy or guidelines. Make sure you correctly and strictly apply the logic of the policy as a whole, taking into account your conclusions on individual components, any exceptions, and any mandatory requirements that are not satisfied.",
    )
    reasons: list[str] = Field(
        ..., description="List of reasoning steps. Each step should comprise one to five sentences of text presenting a clear logical analysis."
    )
    confidence: Literal["high", "medium", "low"] = Field(description="Your confidence in the overall conclusion.")


# --- Judge Agent ---
class Judge(LLMAgent):
    """
    An LLM agent that evaluates content based on criteria using a Buttermilk template/model.
    It expects input via AgentInput and outputs results conforming to AgentReasons.
    Handler methods marked with @buttermilk_handler will be registered by AutogenAgentAdapter.
    """

    # Define the expected Pydantic model for the structured output ('outputs' field)
    _output_model: Optional[Type[BaseModel]] = AgentReasons

    # Initialization is handled by LLMAgent -> Agent -> AgentConfig

    # --- Custom Autogen Handlers ---

    @buttermilk_handler(AgentInput)  # Mark this method to handle AgentInput messages
    async def handle_judge_invocation(
        self,
        message: AgentInput,
        # Autogen context might not be directly available here unless the adapter passes it.
        # The core logic should rely on the 'message' (AgentInput).
        # ctx: MessageContext # Removed for simplicity, adapter handles Autogen context
    ) -> Optional[AgentOutput]:  # Return AgentOutput or None
        """Handles the primary invocation request for the Judge agent."""
        logger.info(f"Judge agent '{self.name}' handling invocation: {message.inputs.get('prompt', 'No prompt')}")

        # Use the _process method inherited from LLMAgent
        # _process already takes AgentInput and returns AgentOutput/ToolOutput/None
        try:
            # Call the LLMAgent's process method
            result = await self._process(inputs=message)  # Pass cancellation token if available/needed

            # _process in LLMAgent is expected to publish results via runtime.
            # If we need to return something specific for Autogen, adjust here.
            # For now, let's assume _process handles publishing and we just return its result.
            if isinstance(result, AgentOutput):
                logger.debug(f"Judge _process completed, returning AgentOutput.")
                return result
            elif result is None:
                logger.debug("Judge _process returned None.")
                return None
            else:
                logger.warning(f"Judge _process returned unexpected type: {type(result)}. Returning None.")
                return None  # Or handle ToolOutput differently if needed

        except Exception as e:
            logger.error(f"Error during Judge handle_judge_invocation: {e}", exc_info=True)
            # Create and return an error AgentOutput
            return AgentOutput(error=[str(e)], inputs=message)

    # Add other handlers as needed, e.g., for different message types
    # @buttermilk_handler(SomeOtherMessageType)
    # async def handle_other_message(self, message: SomeOtherMessageType) -> None:
    #     logger.info(f"Judge handling {type(message)}")
    #     # ... implementation ...
