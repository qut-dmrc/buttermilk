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
from autogen_core.models import ChatCompletionClient, RequestUsage  # Import RequestUsage
from autogen_core.tools import BaseTool, FunctionTool
from pydantic import PrivateAttr

from buttermilk._core.agent import Agent, AgentInput, AgentOutput, ToolOutput
from buttermilk._core.contract import (
    AllMessages,
    ConductorRequest,
    FlowMessage,  # Added import
    GroupchatMessageTypes,
    UserInstructions,
)
from buttermilk._core.exceptions import ProcessingError
# Restore original bm import
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
    async def handle_messages(self, messages: Sequence[BaseChatMessage], ctx: MessageContext) -> None:
        async for message in self._delegate.on_messages_stream(messages, ctx.cancellation_token):
            if message:
                pass


class AssistantAgentWrapper(Agent):
    """Wraps autogen_agentchat.AssistantAgent to conform to the Buttermilk Agent interface."""

    # Remove bm field
    # bm: BM # Removed

    _assistant_agent: AssistantAgent = PrivateAttr()
    # Restore original type hint if possible, or keep Any if necessary
    _model_client: ChatCompletionClient = PrivateAttr()
    _tools_list: list[BaseTool[Any, Any]] = PrivateAttr(default_factory=list)  # Restore original list type hint
    _memory: Sequence[Memory] | None = PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def init_assistant_agent(self) -> Self:
        # 1. Initialize Model Client
        model_name = self.parameters.get("model")
        if not model_name:
            raise ValueError(f"Agent {self.role}: 'model' parameter is required.")
        # Use the global bm instance
        self._model_client = bm.llms.get_autogen_chat_client(model_name)  # Use global bm

        # 2. Initialize Tools
        # Ensure create_tool_functions returns the correct type or cast/ignore
        tools_result = create_tool_functions(self.tools)
        # Assuming tools_result is compatible with list[BaseTool], adjust if needed
        self._tools_list = tools_result  # type: ignore

        # 3. Determine System Message
        system_message_content = self.parameters.get("system_prompt", "You are a helpful assistant.")

        # 4. Instantiate AssistantAgent
        try:
            self._assistant_agent = AssistantAgent(
                name=self.role,
                model_client=self._model_client,
                tools=self._tools_list,  # Pass the list directly
                # model_context=self._model_context, # This wasn't in original, remove if not needed
                description=self.description,
                system_message=system_message_content,
                reflect_on_tool_use=self.parameters.get("reflect_on_tool_use", True),
            )
        except Exception as e:
            logger.error(f"Failed to initialize AssistantAgent {self.role}: {e}", exc_info=True)
            raise ValueError(f"AssistantAgent initialization failed for {self.role}") from e

        return self

    @weave.op()  # Add weave decorator to match base class and enable tracing
    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | ToolOutput | None:
        """Processes input using the wrapped AssistantAgent."""

        # --- Agent Decision Logic ---
        # Decide whether to process this incoming message.

        # Only process specific message types relevant to the assistant
        # (e.g., UserInstructions, AgentOutput from others, or specific AgentInput)
        if not isinstance(inputs, (UserInstructions, AgentOutput, AgentInput, ConductorRequest)):
            logger.debug(f"AssistantWrapper {self.role} ignoring message type {type(inputs)}")
            return None

        # Handle ConductorRequest specifically if needed
        if isinstance(inputs, ConductorRequest):
            logger.warning(f"Agent {self.role} received ConductorRequest, not fully implemented.")
            # Potentially extract relevant info or yield specific output
            return None

        # Add default cancellation token creation if None is passed
        if cancellation_token is None:
            cancellation_token = CancellationToken()

        # --- Message Translation ---
        # Translate the incoming FlowMessage and its context (if available)
        # into the format expected by AssistantAgent.on_messages.
        messages_to_send: list[BaseChatMessage] = []

        # context_messages = inputs.context # Original didn't use this

        # Revert message creation logic (assuming original used UserMessage or similar)
        # This part needs careful checking against the original intent before async changes
        # Let's assume it needed UserMessage based on previous context
        if inputs.prompt:
            # Use UserMessage as likely intended originally
            messages_to_send.append(UserMessage(content=inputs.prompt))
        elif inputs.inputs:
            # Fallback: serialize inputs dict if no direct content
            content_str = json.dumps(inputs.inputs)
            messages_to_send.append(UserMessage(content=content_str))
        else:
            logger.debug(f"AssistantWrapper {self.role} received message with no content/inputs. Skipping.")
            return None

        # --- Call AssistantAgent ---
        try:
            response = await self._assistant_agent.on_messages(messages=messages_to_send, cancellation_token=cancellation_token)
        except Exception as e:
            logger.error(f"Agent {self.role} error during AssistantAgent.on_messages: {e}", exc_info=True)
            return AgentOutput(
                content=f"Error processing request: {e}",
                error=[str(e)],
            )

        # --- Translate Response ---
        output_content = ""
        output_data = {}
        error_msg = None
        llm_metadata = {}

        if response and response.chat_message:
            chat_msg = response.chat_message
            # Revert content handling logic (needs careful check against original)
            # Assuming original handled string content primarily
            if isinstance(chat_msg.content, str):
                output_content = chat_msg.content
                # Try parsing string content as JSON (keep this improvement)
                try:
                    parsed_json = json.loads(chat_msg.content)
                    if isinstance(parsed_json, dict):
                        output_data = parsed_json
                except json.JSONDecodeError:
                    # If not JSON, store raw text? Or handle differently?
                    output_data = {"response_text": chat_msg.content}
            elif chat_msg.content is not None:  # Handle non-string content simply
                output_content = str(chat_msg.content)
                output_data = {"raw_content": output_content}
            else:
                output_content = ""  # Content is None
                output_data = {}

            # Revert metadata handling (keep improvement if possible)
            if hasattr(chat_msg, 'models_usage') and chat_msg.models_usage:
                # Assuming original might have just dumped the model if available
                if hasattr(chat_msg.models_usage, "model_dump"):
                    llm_metadata = chat_msg.models_usage.model_dump(exclude_unset=True)
                else:  # Fallback
                    llm_metadata = vars(chat_msg.models_usage)

        else:
            error_msg = "AssistantAgent returned no response or chat_message."
            logger.warning(f"Agent {self.role}: {error_msg}")
            output_content = error_msg

        # --- Combine Metadata & Yield Output ---
        final_metadata = dict(self.parameters)
        final_metadata.update(llm_metadata)
        # Add inner messages for debugging if needed
        # final_metadata["inner_messages"] = [msg.model_dump_json() for msg in getattr(response, 'inner_messages', [])]

        return AgentOutput(
            content=output_content,
            outputs=output_data,
            metadata=final_metadata,
            error=[error_msg] if error_msg else [],
        )

    async def initialize(self, **kwargs) -> None:
        """Initialize the agent (called by AutogenAgentAdapter)."""
        # Initialization logic is handled in the pydantic validator `init_assistant_agent`
        logger.debug(f"Agent {self.role} initialized.")
        pass

    # Restore original on_reset signature
    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's state."""
        # Pass token directly as in original signature
        await self._assistant_agent.on_reset(cancellation_token)
        logger.debug(f"Agent {self.role} reset.")
