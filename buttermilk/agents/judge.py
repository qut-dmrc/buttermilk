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


# Keep AgentRequest if it defines the expected input structure for this agent's task
class AgentRequest(BaseModel):
    prompt: str = Field(..., description="The core question or instruction for the agent.")
    # Records might be passed differently in a group chat context, perhaps initially or via context.
    # Let's assume they are part of the initial setup or context for now.
    # records: list[Record] = Field(default_factory=list, description="Supporting records for analysis.")


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

    # Add __init__ to handle arguments for both parent classes
    def __init__(
        self,
        name: str,  # Required by RoutedAgent
        parameters: Dict[str, Any],  # Required by LLMAgent logic
        tools: List[str] = [],  # Optional tools for LLMAgent
        fail_on_unfilled_parameters: bool = True,  # LLMAgent config
        **kwargs,  # Pass other args to RoutedAgent (e.g., description, system_message)
    ):
        # Initialize LLMAgent parts (handled by Pydantic validators via super().__init__)
        # Initialize RoutedAgent parts
        RoutedAgent.__init__(self, name=name, **kwargs)
        # Manually set LLMAgent fields after RoutedAgent init, as LLMAgent doesn't have a standard __init__
        self.parameters = parameters
        self.tools = tools
        self.fail_on_unfilled_parameters = fail_on_unfilled_parameters
        # Trigger LLMAgent's Pydantic validators manually if needed, or rely on them being called implicitly
        # Note: This assumes LLMAgent's validators run correctly even when it's not the first parent.
        # If issues arise, might need to call self.model_post_init(None) or similar.

        # Register the message handler
        self.register_handler(self.handle_groupchat_message)

    # Use model_validator from Pydantic for post-init logic like LLMAgent
    @model_validator(mode="after")
    def _init_llmagent_parts(self) -> Self:
        # Explicitly call LLMAgent's validators if they weren't triggered automatically
        # This ensures _model_client and _tools_list are initialized
        super(LLMAgent, self)._init_model()  # Call LLMAgent's model init
        super(LLMAgent, self)._load_tools()  # Call LLMAgent's tool loading
        return self

    @message_handler
    async def handle_groupchat_message(
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
