import asyncio
from curses import meta
import json
import pprint
from types import NoneType
from typing import Any, AsyncGenerator, Callable, Literal, Optional, Self, Dict, List, Union

from autogen_core.models._types import UserMessage
import pydantic
import regex as re

# Make sure to import necessary Autogen components
from autogen_core import CancellationToken, FunctionCall, MessageContext, RoutedAgent, message_handler, Agent as AutogenAgent
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    # Message, # Removed incorrect import
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema, ToolConfig  # Import ToolConfig if needed later
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import weave

# Correct the import from _core.agent
from buttermilk._core.agent import AgentInput, AgentOutput
from buttermilk.agents.llm import LLMAgent  # Import LLMAgent directly
from buttermilk._core.contract import (
    AllMessages,  # Keep relevant contract types if needed
    LLMMessage,  # Ensure LLMMessage is imported if used for context conversion
    ConductorRequest,
    FlowMessage,
    GroupchatMessageTypes,
    LLMMessage,
    ToolOutput,
    UserInstructions,
    TaskProcessingComplete,  # Keep relevant contract types if needed
    ProceedToNextTaskSignal,  # Keep relevant contract types if needed
    OOBMessages,  # Keep relevant contract types if needed
)
from buttermilk._core.types import Record
from buttermilk.bm import bm, logger


# Keep AgentRequest if it defines the expected input structure for this agent's task
class AgentRequest(BaseModel):
    prompt: str = Field(..., description="The core question or instruction for the agent.")
    records: list[Record] = Field(default_factory=list, description="Supporting records for analysis.")


class AgentReasons(BaseModel):
    conclusion: str = Field(..., description="Your conlusion or final answer.")
    prediction: bool = Field(
        description="True if the content violates the policy or guidelines. Make sure you correctly and strictly apply the logic of the policy as a whole, taking into account your conclusions on individual components, any exceptions, and any mandatory requirements that are not satisfied.",
    )
    reasons: list[str] = Field(
        ..., description="List of reasoning steps. Each step should comprise one to five sentences of text presenting a clear logical analysis."
    )
    confidence: Literal["high", "medium", "low"] = Field(description="Your confidence in the overall conclusion.")


# Inherit from LLMAgent (for Buttermilk features) and RoutedAgent (for Autogen group chat)
class Judge(LLMAgent, RoutedAgent):
    """
    An LLM agent that uses a Buttermilk template/model configuration
    and participates in an Autogen group chat as a RoutedAgent.
    It expects input conforming to AgentRequest (primarily the prompt)
    and outputs results conforming to AgentReasons.
    """

    _output_model: Optional[type[BaseModel]] = AgentReasons  # Define the expected structured output

    def __init__(
        self,
        name: str,  # Required by ConversableAgent (parent of RoutedAgent)
        parameters: Dict[str, Any],  # Required by LLMAgent logic
        tools: List[str] = [],  # Optional tools for LLMAgent (list of names)
        fail_on_unfilled_parameters: bool = True,  # LLMAgent config
        **kwargs,  # Pass other args to RoutedAgent (e.g., description, system_message, llm_config)
    ):
        # Initialize RoutedAgent first (passes name positionally to ConversableAgent)
        RoutedAgent.__init__(self, name, **kwargs)

        # Set LLMAgent fields. Pydantic should handle validation via model_validator.
        # We don't assign self.tools here directly to avoid type conflicts if RoutedAgent has a 'tools' attribute.
        # LLMAgent's validator will handle the 'tools' list provided in parameters if needed, or load from self.tools.
        self.parameters = parameters
        # Store the tool names list separately if needed, or ensure LLMAgent loads it correctly
        self._llm_agent_tool_names = tools  # Store separately to avoid conflict
        self.fail_on_unfilled_parameters = fail_on_unfilled_parameters

    @message_handler
    async def handle_groupchat_message(
        self,
        message: AgentInput,
        ctx: MessageContext,
    ) -> Optional[AgentReasons]:
        """Handles incoming messages in the Autogen group chat."""
        logger.info(f"Judge '{self.name}' received message from '{sender.name}' in topic '{ctx.topic_id}': {message}")

        # Use the _process method inherited from LLMAgent
        result: AgentReasons = await self._process(inputs=message)

        if result:
            # Publish the structured output back to the group chat
            # Autogen expects a dict or string usually. Convert Pydantic model.
            await self.runtime.publish_message(message=response_data.model_dump(), topic_id=ctx.topic_id, sender=self.id)  # Send as dict
            return result
        logger.warning(f"Judge '{self.name}' did not produce a publishable output.")
        # Handle cases where no output was generated
        return None  # No reply
