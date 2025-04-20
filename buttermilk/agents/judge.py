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


# Input/Output models remain the same
class AgentRequest(BaseModel):
    prompt: str = Field(..., description="The core question or instruction for the agent.")
    # records: list[Record] = Field(default_factory=list, description="Supporting records for analysis.") # Records are handled by AgentInput


class AgentReasons(BaseModel):
    conclusion: str = Field(..., description="Your conclusion or final answer.")
    prediction: bool = Field(
        description="True if the content violates the policy or guidelines. Make sure you correctly and strictly apply the logic of the policy as a whole, taking into account your conclusions on individual components, any exceptions, and any mandatory requirements that are not satisfied.",
    )
    reasons: list[str] = Field(
        ..., description="List of reasoning steps. Each step should comprise one to five sentences of text presenting a clear logical analysis."
    )
    confidence: Literal["high", "medium", "low"] = Field(description="Your confidence in the overall conclusion.")


from buttermilk.libs.autogen import AutogenRoutedMixin  # Import the new mixin


# Inherit from LLMAgent (for Buttermilk/Pydantic features) and AutogenRoutedMixin (for Autogen features)
# Order matters: Pydantic models usually need to be earlier in MRO.
class Judge(LLMAgent, AutogenRoutedMixin):
    """
    An LLM agent that uses a Buttermilk template/model configuration (via LLMAgent)
    and participates in an Autogen group chat (via AutogenRoutedMixin).
    It expects input conforming to AgentInput (handled by LLMAgent/Agent base)
    and aims to output results conforming to AgentReasons.
    """

    # Define the expected structured output model for LLMAgent's processing
    _output_model: Optional[type[BaseModel]] = AgentReasons

    def __init__(
        self,
        # --- Arguments for LLMAgent/AgentConfig (handled by Pydantic) ---
        role: str = "judge",  # Default role for this agent type
        parameters: Dict[str, Any] = Field(default_factory=dict),  # Model, template etc.
        tools: List[ToolConfig] = Field(default_factory=list),  # Buttermilk ToolConfig
        fail_on_unfilled_parameters: bool = True,
        # Add other AgentConfig fields if needed (id, inputs, outputs etc.)
        # --- Arguments specifically for RoutedAgent (required by AutogenRoutedMixin) ---
        name: str = "Judge",  # Autogen requires a name
        description: str = "Evaluates content based on criteria.",  # Default description
        system_message: Optional[str] = None,  # Optional system message for Autogen
        # --- Capture remaining kwargs ---
        **kwargs,  # Captures other AgentConfig fields AND RoutedAgent fields
    ):
        """
        Initializes the Judge agent.

        Args:
            role: The Buttermilk role.
            parameters: Configuration for the LLM, template, etc. (for LLMAgent).
            tools: List of tools the agent can use (Buttermilk format).
            fail_on_unfilled_parameters: LLMAgent setting.
            name: The name for the Autogen agent.
            description: Description for the Autogen agent.
            system_message: System message for the Autogen agent.
            **kwargs: Additional arguments for AgentConfig (like 'id', 'inputs', 'outputs')
                      and potentially other RoutedAgent parameters.
        """
        # 1. Initialize Pydantic part (LLMAgent -> Agent -> AgentConfig)
        #    Pydantic handles collecting args matching its fields from kwargs.
        #    We pass all relevant args explicitly or via kwargs.
        #    `name` is also an AgentConfig field, so it's handled here too.
        LLMAgent.__init__(
            self,
            role=role,
            name=name,  # Pass name to AgentConfig as well
            description=description,  # Pass description to AgentConfig
            parameters=parameters,
            tools=tools,
            fail_on_unfilled_parameters=fail_on_unfilled_parameters,
            **kwargs,  # Pass remaining AgentConfig fields (id, inputs, outputs etc.)
            # and potentially non-AgentConfig RoutedAgent fields too
        )

        # 2. Initialize Autogen RoutedAgent part via the Mixin
        #    We need to explicitly call the superclass __init__ of the *mixin's* parent.
        #    Pass only the arguments relevant to RoutedAgent.
        #    Pydantic's __init__ already handled shared fields like 'name', 'description'.
        #    We only need to pass fields *specifically* for RoutedAgent if any beyond 'name'.
        #    Common RoutedAgent args: name, description, system_message, llm_client, etc.
        #    The mixin itself doesn't have an __init__, so we call RoutedAgent's directly.
        RoutedAgent.__init__(
            self,
            name=name,  # Must be passed
            description=description,  # Recommended
            system_message=system_message,  # Optional
            # Pass any other specific RoutedAgent kwargs if they were captured in **kwargs
            # Be careful not to pass AgentConfig fields here again.
            **{k: v for k, v in kwargs.items() if k in RoutedAgent.model_fields},  # Pass only valid RoutedAgent fields
        )

        # Pydantic's model validators in LLMAgent (init_model, _load_tools)
        # should run automatically as part of the LLMAgent.__init__ process.
        # No need for _init_llmagent_parts validator here.

        # Message handlers are registered automatically by @message_handler in the mixin.
        # No need for self.register_handler(...)

    # Remove the old message handler, the mixin provides it now.
    # @message_handler
    # async def handle_groupchat_message(...) -> ... :
    #    ...
