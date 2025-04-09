import asyncio
import json
from typing import Any, AsyncGenerator, Self, Sequence

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    BaseChatMessage,
    
    TextMessage,
)
from autogen_core.models._types import UserMessage, AssistantMessage
from autogen_core import CancellationToken, FunctionCall
from autogen_core.memory import Memory
from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool, FunctionTool
from pydantic import PrivateAttr

from buttermilk._core.agent import Agent, AgentInput, AgentOutput
from buttermilk._core.contract import (
    AllMessages,
    ConductorRequest,
    FlowMessage, # Added import
    GroupchatMessageTypes,
    UserInstructions,
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk.bm import bm, logger
from buttermilk.utils._tools import create_tool_functions

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler

class SimpleAutogenChatWrapper(RoutedAgent):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._model_client: ChatCompletionClient = PrivateAttr()
        self._delegate = AssistantAgent(name, model_client=self._model_client)
    
    @message_handler
    async def handle_messages(self, messages: Sequence[BaseChatMessage],  ctx: MessageContext) -> None:
        async for message in self._delegate.on_messages_stream(messages, ctx.cancellation_token):
            if message:
                pass

class AssistantAgentWrapper(Agent):
    """Wraps autogen_agentchat.AssistantAgent to conform to the Buttermilk Agent interface."""

    _assistant_agent: AssistantAgent = PrivateAttr()
    _model_client: ChatCompletionClient = PrivateAttr()
    _tools_list: list[BaseTool[Any, Any]] = PrivateAttr(default_factory=list)
    _memory: Sequence[Memory] | None = PrivateAttr(default=None) # Add memory if needed later

    @pydantic.model_validator(mode="after")
    def init_assistant_agent(self) -> Self:
        # 1. Initialize Model Client
        model_name = self.parameters.get("model")
        if not model_name:
            raise ValueError(f"Agent {self.id}: 'model' parameter is required.")
        self._model_client = bm.llms.get_autogen_chat_client(model_name)

        # 2. Initialize Tools
        # self._tools_list = create_tool_functions(self.tools)

        # 3. Determine System Message
        system_message_content = self.parameters.get("system_prompt", "You are a helpful assistant.")

        # 4. Instantiate AssistantAgent
        try:
            self._assistant_agent = AssistantAgent(
                name=self.id,
                model_client=self._model_client,
                tools=self._tools_list,
                model_context=self._model_context,
                description=self.description,
                system_message=system_message_content,
                reflect_on_tool_use=self.parameters.get("reflect_on_tool_use", True),
            )
        except Exception as e:
            logger.error(f"Failed to initialize AssistantAgent {self.id}: {e}", exc_info=True)
            raise ValueError(f"AssistantAgent initialization failed for {self.id}") from e

        return self

    async def _process(
        self,
        inputs: AgentInput,cancellation_token: CancellationToken,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput | None, None]:
        """Processes input using the wrapped AssistantAgent."""

        # --- Agent Decision Logic ---
        # Decide whether to process this incoming message.
        # Don't respond to own messages or irrelevant types.
        if inputs.source == self.id:
            yield None
            return

        # Only process specific message types relevant to the assistant
        # (e.g., UserInstructions, AgentOutput from others, or specific AgentInput)
        if not isinstance(message, (UserInstructions, AgentOutput, AgentInput, ConductorRequest)):
             logger.debug(f"AssistantWrapper {self.id} ignoring message type {type(message)} from {message.source}")
             yield None
             return

        # Handle ConductorRequest specifically if needed
        if isinstance(message, ConductorRequest):
             logger.warning(f"Agent {self.id} received ConductorRequest, not fully implemented.")
             # Potentially extract relevant info or yield specific output
             yield None
             return

        if cancellation_token is None:
            cancellation_token = CancellationToken()

        # --- Message Translation ---
        # Translate the incoming FlowMessage and its context (if available)
        # into the format expected by AssistantAgent.on_messages.
        messages_to_send: list[BaseChatMessage] = []

        context_messages = message.context
        for ctx_msg in context_messages:
             # Map Buttermilk message types to Autogen message types
             if isinstance(ctx_msg, AgentOutput) and ctx_msg.source == self.id:
                 messages_to_send.append(AssistantMessage(content=str(ctx_msg.content), source=ctx_msg.source))
             elif isinstance(ctx_msg, (UserInstructions, AgentInput, AgentOutput)):
                 messages_to_send.append(UserMessage(content=str(ctx_msg.content), source=getattr(ctx_msg, 'agent_id', 'unknown')))
             # Add mappings for other types if necessary

        # Add the current incoming message content as the latest UserMessage
        if message.content:
            messages_to_send.append(UserMessage(content=message.content, source=message.source))
        elif message.inputs:
             # Fallback: serialize inputs dict if no direct content
             messages_to_send.append(UserMessage(content=json.dumps(message.inputs), source=message.source))
        else:
             # If the message has no content or inputs, maybe don't process?
             logger.debug(f"AssistantWrapper {self.id} received message with no content/inputs from {message.source}. Skipping.")
             yield None
             return

        # --- Call AssistantAgent ---
        try:
            response = await self._assistant_agent.on_messages(
                messages=messages_to_send, cancellation_token=cancellation_token
            )
        except Exception as e:
            logger.error(f"Agent {self.id} error during AssistantAgent.on_messages: {e}", exc_info=True)
            yield AgentOutput(
                source=self.id,
                role=self.role,
                content=f"Error processing request: {e}",
                error=[str(e)],
                records=message.records, # Pass through records from input
            )
            return

        # --- Translate Response ---
        output_content = ""
        output_data = {}
        error_msg = None
        llm_metadata = {}

        if response and response.chat_message:
            chat_msg = response.chat_message
            # Handle various content types (str, list, BaseModel)
            if isinstance(chat_msg.content, str):
                 output_content = chat_msg.content
                 # Try parsing string content as JSON
                 try:
                     parsed_json = json.loads(chat_msg.content)
                     if isinstance(parsed_json, dict):
                         output_data = parsed_json
                 except json.JSONDecodeError:
                     output_data = {"response_text": chat_msg.content}
            elif isinstance(chat_msg.content, list): # Often tool calls or structured output list
                 output_content = json.dumps([item.model_dump() if hasattr(item, 'model_dump') else item for item in chat_msg.content], indent=2)
                 output_data = {"structured_list": chat_msg.content} # Or process list items further
            elif hasattr(chat_msg.content, 'model_dump'): # Pydantic model
                 output_data = chat_msg.content.model_dump()
                 output_content = json.dumps(output_data, indent=2)
            else: # Fallback
                 output_content = str(chat_msg.content)
                 output_data = {"raw_content": output_content}


            if hasattr(chat_msg, 'models_usage') and chat_msg.models_usage:
                 llm_metadata = chat_msg.models_usage.model_dump(exclude_unset=True)

        else:
            error_msg = "AssistantAgent returned no response or chat_message."
            logger.warning(f"Agent {self.id}: {error_msg}")
            output_content = error_msg

        # --- Combine Metadata & Yield Output ---
        final_metadata = dict(self.parameters)
        final_metadata.update(llm_metadata)
        # Add inner messages for debugging if needed
        # final_metadata["inner_messages"] = [msg.model_dump_json() for msg in getattr(response, 'inner_messages', [])]

        yield AgentOutput(
            source=self.id,
            role=self.role,
            content=output_content,
            outputs=output_data,
            metadata=final_metadata,
            records=message.records, # Pass through records from input
            error=[error_msg] if error_msg else [],
        )

    async def initialize(self, **kwargs) -> None:
        """Initialize the agent (called by AutogenAgentAdapter)."""
        # Initialization logic is handled in the pydantic validator `init_assistant_agent`
        logger.debug(f"Agent {self.id} initialized.")
        pass

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None: # Added optional token
        """Reset the agent's state."""
        await self._assistant_agent.on_reset(cancellation_token)
        logger.debug(f"Agent {self.id} reset.")
