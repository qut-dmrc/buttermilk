from __future__ import annotations  # Add this import at the very top

import asyncio
from collections.abc import Awaitable, Callable

from typing import Any, Optional, Union, Sequence, TYPE_CHECKING, List, Dict, Callable, Awaitable

from buttermilk._core.contract import TaskProcessingStarted, TaskProcessingComplete
from buttermilk._core.contract import ConductorResponse
from autogen_core import (
    DefaultTopicId,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)

from buttermilk._core.agent import Agent, AgentConfig, ToolOutput
from buttermilk._core.contract import (
    CONDUCTOR,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    GroupchatMessageTypes,
    HeartBeat,
    ManagerMessage,
    FlowMessage,
    ManagerRequest,
    OOBMessages,
    TaskProcessingComplete,
    TaskProcessingStarted,
    ToolOutput,  # Added ToolOutput here as well
)


from buttermilk.bm import logger

# Keep Agent import for type hints if needed, but ToolConfig not used here
from buttermilk._core.agent import Agent
import asyncio


class AutogenRoutedMixin(RoutedAgent):
    """
    Mixin class to integrate Buttermilk agent capabilities with Autogen's RoutedAgent.

    This mixin should be inherited alongside a Buttermilk Agent subclass (e.g., LLMAgent).
    It provides message handlers that delegate processing to the Buttermilk agent's
    methods (`__call__`, `_listen`, `_handle_events`).
    """

    # --- Expected Attributes from Buttermilk Agent ---
    # These are placeholders for the type checker.
    role: str
    description: str

    # Ensure __call__ is expected (all Agents should be callable)
    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...

    _heartbeat: asyncio.Queue
    _listen: Callable[..., Awaitable[None]]
    _handle_events: Callable[..., Awaitable[Optional[Any]]]

    # --- End Expected Attributes ---

    # --- Initialization ---
    # The actual Buttermilk Agent initialization happens in the inheriting class (e.g., Judge).
    # This mixin relies on the inheriting class to also initialize RoutedAgent correctly.
    # See the Judge agent's __init__ for an example.
    # --- End Initialization ---

    # --- Message Handlers ---

    @message_handler
    async def _handle_heartbeat_mixin(self, message: HeartBeat, ctx: MessageContext) -> None:
        """Handle heartbeat messages by adding to the agent's internal queue."""
        # Now self._heartbeat and self.name should be recognized by the type checker
        try:
            self._heartbeat.put_nowait(message.go_next)
        except asyncio.QueueFull:
            logger.debug(f"Heartbeat failed, agent {self.name} is idle or running behind.")
        except AttributeError:
            # This might still happen at runtime if the inheriting class doesn't provide it,
            # but the static type checker should be happier.
            logger.error(f"Agent {self.name} missing expected _heartbeat attribute at runtime.")

    @message_handler
    async def handle_invocation_mixin(
        self,
        message: AgentInput,
        ctx: MessageContext,
    ) -> None:
        """
        Handle direct invocation requests (AgentInput).
        Delegates to the Buttermilk agent's __call__ method.
        Publishes start/complete signals and the agent's response.
        """
        # self is callable and has name/id/role due to placeholder hints
        topic_id = DefaultTopicId(type=self.role)

        # Cast self.id to string if it's not already str type (Autogen ID)
        # Buttermilk Agent ID might be different, use Autogen's self.id here
        agent_id_str = str(self.id)

        await self.publish_message(TaskProcessingStarted(agent_id=agent_id_str, role=self.role, task_index=0), topic_id=topic_id)

        # Delegate to the Buttermilk Agent's main call logic
        # Simplify hint within method body for Pylance
        response: Optional[Any] = await self(
            message=message,
            cancellation_token=ctx.cancellation_token,
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )

        # Publish the response(s) if any
        if response:
            if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):  # More robust check for sequence
                for res_msg in response:
                    await self.publish_message(res_msg, topic_id=topic_id)
            else:  # Handle single message ("AgentOutput", "ToolOutput", "ErrorEvent", etc.)
                await self.publish_message(response, topic_id=topic_id)

        # Determine if there was an error for the completion message
        is_error = False
        # Import types needed for runtime isinstance checks
        from buttermilk._core.contract import AgentOutput, ErrorEvent

        if isinstance(response, AgentOutput) and response.error:
            is_error = True
        elif isinstance(response, ErrorEvent):
            is_error = True
        # Add other potential error indicators if necessary

        await self.publish_message(
            TaskProcessingComplete(agent_id=agent_id_str, role=self.role, task_index=0, more_tasks_remain=False, is_error=is_error),
            topic_id=topic_id,
        )
        # RoutedAgent handlers typically don't return values directly; they publish.

    @message_handler
    async def handle_groupchat_message_mixin(
        self,
        message: "GroupchatMessageTypes",  # Use string forward reference
        ctx: MessageContext,
    ) -> None:
        """
        Handle incoming group messages by delegating to the Buttermilk agent's _listen method.
        """
        # self._listen and self.name/role are known via placeholders
        public_topic_id = DefaultTopicId(type=self.role)
        private_topic_id = ctx.topic_id  # Can be None, handle in callback maker

        # Use the placeholder hint for _listen
        await self._listen(
            message=message,
            cancellation_token=ctx.cancellation_token,
            public_callback=self._make_publish_callback(topic_id=public_topic_id),
            message_callback=self._make_publish_callback(topic_id=private_topic_id),  # Pass potentially None topic_id
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )
        # Pylance might still warn about await if _listen hint isn't specific enough, but it should work.

    @message_handler
    async def handle_conductor_request_mixin(
        self,
        message: "ConductorRequest",  # Use string forward reference
        ctx: MessageContext,
    ) -> Optional["ConductorResponse"]:  # Use string forward reference and Optional
        """
        Handle private conductor requests by delegating to the Buttermilk agent's __call__.
        Returns the response directly to the conductor.
        """
        # self is callable and has name/role due to placeholders
        public_topic_id = DefaultTopicId(type=self.role)
        private_topic_id = ctx.topic_id  # Can be None

        # Simplify hint within method body for Pylance
        raw_output: Optional[Any] = await self(
            message=message,
            cancellation_token=ctx.cancellation_token,
            public_callback=self._make_publish_callback(topic_id=public_topic_id),
            message_callback=self._make_publish_callback(topic_id=private_topic_id),
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )

        if isinstance(raw_output, ConductorResponse):
            return raw_output  # Return the ConductorResponse
        elif raw_output is not None:
            # Log if the agent returned something, but not the expected type
            logger.warning(f"Agent {self.name} returned unexpected type {type(raw_output)} for ConductorRequest. Expected ConductorResponse or None.")
        return None  # Return None otherwise

    @message_handler
    async def handle_control_message_mixin(
        self,
        message: OOBMessages,  # Use string forward reference
        ctx: MessageContext,
    ) -> Optional[OOBMessages]:  # Use string forward reference and Optional
        """
        Handle out-of-band control messages by delegating to the Buttermilk agent's _handle_events.
        Returns the response directly.
        """
        # Simplify hint within method body for Pylance
        response: Optional[Any] = await self._handle_events(
            message=message,
            cancellation_token=ctx.cancellation_token,
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )
        # Return response directly if it's not None, relying on _handle_events signature
        if response is not None:
            return response
        return None

    # --- Helper Methods ---

    def _make_publish_callback(self, topic_id: Optional[TopicId]) -> Callable[[FlowMessage], Awaitable[None]]:
        """Create a callback for publishing messages via Autogen runtime. Handles optional topic_id."""

        async def publish_callback(message: FlowMessage) -> None:
            """Callback function to publish a Buttermilk message via Autogen."""
            # Only publish if topic_id is valid
            if topic_id is not None:
                await self.publish_message(message, topic_id=topic_id)
            else:
                # Decide how to handle missing topic_id - log? Drop? Use default?
                logger.warning(f"Attempted to publish message via callback with None topic_id for agent {self.name}. Message dropped: {message}")

        return publish_callback
