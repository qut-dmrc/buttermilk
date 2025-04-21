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
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import weave

# Correct the import from _core.agent
from buttermilk._core.agent import AgentInput, AgentOutput
from buttermilk.agents.llm import LLMAgent  # Import LLMAgent directly
from buttermilk._core.contract import (
    AllMessages,  # Keep relevant contract types if needed
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
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import AutoGenWrapper, CreateResult
from buttermilk._core.types import Record
from buttermilk.bm import bm, logger

# Remove unused imports if any, keep necessary ones
# from buttermilk.utils._tools import create_tool_functions # LLMAgent handles this
# from buttermilk.utils.json_parser import ChatParser # LLMAgent handles this
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
    make_messages,  # LLMAgent handles this
)


# --- Pydantic Models ---


class AgentReasons(BaseModel):
    conclusion: str = Field(..., description="Your conlusion or final answer.")
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
    async def handle_agent_input(
        self,
        message: AgentInput,
        ctx: MessageContext,
    ) -> Optional[AgentReasons]:
        """Handles incoming messages in the Autogen group chat."""
        logger.info(f"Judge '{self.name}' received message from '{sender.name}' in topic '{ctx.topic_id}': {message}")

        # Use the _process method inherited from LLMAgent
        result: AgentOutput = await self._process(inputs=message)
        # Publish the structured output back to the group chat
        # Autogen expects a dict or string usually. Convert Pydantic model.
        await self.runtime.publish_message(message=response_data.model_dump(), topic_id=ctx.topic_id, sender=self.id)  # Send as dict

        logger.warning(f"Judge '{self.name}' did not produce a publishable output.")
        # Handle cases where no output was generated
        return None  # No reply
