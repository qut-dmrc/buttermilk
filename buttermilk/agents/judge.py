"""
Defines the Judge agent, an LLM-based agent specialized for evaluating content
against predefined criteria.
"""

import asyncio
from curses import meta
import json
import pprint
from types import NoneType
from typing import Any, AsyncGenerator, Callable, Literal, Optional, Self, Dict, List, Type, Union

# Import Autogen core components needed for type hints and potential interaction (though handled by adapter)
from autogen_core import CancellationToken, FunctionCall, MessageContext, RoutedAgent, message_handler, Agent as AutogenAgent
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,  # Used for type hinting in context
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema
import pydantic
from promptflow.core._prompty_utils import parse_chat  # Used in templating utils
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import regex as re
import weave  # Likely for logging/tracing

# Buttermilk core imports
from buttermilk._core.agent import AgentInput, AgentOutput, buttermilk_handler  # Base types and decorator
from buttermilk._core.contract import (
    # TODO: Review if all these contract types are strictly necessary for this agent's logic vs. type hinting context.
    AllMessages,
    ConductorRequest,
    FlowMessage,
    GroupchatMessageTypes,
    LLMMessage,
    OOBMessages,
    ProceedToNextTaskSignal,
    TaskProcessingComplete,
    ToolOutput,
    UserInstructions,
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import AutoGenWrapper, CreateResult  # LLM interaction wrappers
from buttermilk._core.types import Record  # Data record structure
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents
from buttermilk.bm import bm, logger  # Global Buttermilk instance and logger

# Utility imports
# TODO: Confirm these utils are necessary here or only in LLMAgent/templating.py
from buttermilk.utils.templating import (
    _parse_prompty,  # Used if loading Prompty files directly here
    load_template,  # Used if loading Jinja templates directly here
    make_messages,  # Handled by LLMAgent
)


# --- Pydantic Models ---


class AgentReasons(BaseModel):
    """
    Structured output model for the Judge agent's evaluation.

    Defines the expected JSON structure returned by the LLM after evaluating
    content against the provided criteria.
    """

    conclusion: str = Field(..., description="Your conclusion or final answer summarizing the evaluation.")
    prediction: bool = Field(
        ...,  # Make field required
        description="Boolean flag indicating if the content violates the policy/guidelines. This should be derived logically from the reasoning and criteria application.",
    )
    reasons: list[str] = Field(
        ...,
        description="A list of strings, where each string represents a distinct step in the reasoning process leading to the conclusion and prediction.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="The agent's confidence level (high, medium, or low) in its overall conclusion and prediction."  # Make field required
    )

    def __str__(self):
        """Returns a nicely formatted string representation of the evaluation."""
        reasons_str = "\n".join(f"- {reason}" for reason in self.reasons)
        return (
            f"**Conclusion:** {self.conclusion}\n"
            f"**Violates Policy:** {'Yes' if self.prediction else 'No'}\n"
            f"**Confidence:** {self.confidence.capitalize()}\n"
            f"**Reasoning:**\n{reasons_str}"
        )


# --- Judge Agent ---
class Judge(LLMAgent):
    """
    An LLM agent specialized in evaluating content based on provided criteria.

    Inherits from `LLMAgent`, leveraging its capabilities for LLM interaction,
    prompt templating, and structured output parsing. The `Judge` agent is
    configured with a specific prompt template (likely focused on evaluation)
    and expects the LLM to return results conforming to the `AgentReasons` model.

    The `buttermilk_handler` decorator registers methods to handle specific message
    types within the Buttermilk/Autogen ecosystem.
    """

    # Specifies that the expected structured output from the LLM should conform to AgentReasons.
    # LLMAgent's _process method will attempt to parse the LLM response into this model.
    _output_model: Optional[Type[BaseModel]] = AgentReasons

    # Initialization (`__init__`) is handled by the parent LLMAgent, which takes
    # configuration (like model, template, parameters) via its AgentConfig.

    # --- Buttermilk/Autogen Message Handler ---

    # This decorator registers the method with the Buttermilk framework.
    # When this agent (wrapped by AutogenAgentAdapter) receives an AgentInput message
    # via Autogen, the adapter will likely route it to this handler.
    @buttermilk_handler(AgentInput)
    async def evaluate_content(  # Renamed for clarity from handle_agent_input
        self,
        message: AgentInput,
    ) -> AgentOutput:
        """
        Handles an AgentInput request to evaluate content based on the agent's configured criteria.

        Args:
            message: The AgentInput message containing the content/prompt to evaluate.

        Returns:
            An AgentOutput message containing the structured evaluation (AgentReasons)
            or an error if processing fails.
        """
        logger.debug(f"Judge agent '{self.id}' received evaluation request.")
        try:
            # Delegate the core LLM call and output parsing to the parent LLMAgent's _process method.
            # This method handles template rendering, API calls, retries, and parsing into _output_model.
            result: AgentOutput = await self._process(message=message)

            # LLMAgent._process already creates and returns AgentOutput.
            # We might want to add specific logging or post-processing here if needed.
            if not result.is_error and isinstance(result.outputs, AgentReasons):
                logger.debug(f"Judge '{self.id}' completed evaluation successfully.")
            elif not result.is_error:
                logger.warning(f"Judge '{self.id}' completed but output type is not AgentReasons: {type(result.outputs)}")
            else:
                logger.error(f"Judge '{self.id}' encountered an error during processing: {result.outputs}")

            # LLMAgent._process returns the AgentOutput directly.
            # The AutogenAgentAdapter is responsible for publishing this if needed.
            # The commented-out publish line below seems redundant if the adapter handles it.
            # TODO: Verify if explicit publishing is needed here or handled by the adapter/orchestrator.
            # await self._runtime.publish_message(message=result, topic_id=ctx.topic_id, sender=self.id)

            return result

        except Exception as e:
            logger.exception(f"Unexpected error in Judge.evaluate_content for agent '{self.id}': {e}")
            # Create an AgentOutput indicating an error.
            error_output = AgentOutput(agent_id=self.id)
            error_output.set_error(f"Unexpected error in Judge agent: {e}")
            return error_output

    # Note: Other handlers (like _listen, _handle_events) can be added here if the Judge
    # needs to react to other message types or perform background tasks, inheriting or
    # overriding behavior from Agent/LLMAgent as needed.
