from __future__ import annotations  # Add this import at the very top

import asyncio
import inspect  # Import inspect for introspection
from functools import partial  # Import partial for creating wrapper methods
from collections.abc import Awaitable, Callable
from typing import Any, Optional, Union, Sequence, TYPE_CHECKING, List, Dict, Callable, Awaitable, Type, ClassVar  # Added Type, Dict, ClassVar
from typing_extensions import Self  # Import Self type
from dataclasses import dataclass

from autogen_core import (
    DefaultTopicId,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId,
    AgentInstantiationContext,
    AgentRuntime,
    AgentType,
    message_handler,
    # message_handler is not used directly here, registration is manual
)
from buttermilk._core.contract import GroupchatMessageTypes

# Use TYPE_CHECKING to avoid circular imports if Agent needs types from here
if TYPE_CHECKING:
    from buttermilk._core.agent import Agent, AgentConfig
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
        ManagerResponse,
        OOBMessages,
        TaskProcessingComplete,
        TaskProcessingStarted,
        UserInstructions,
        AllMessages,
        ToolOutput,
    )

from buttermilk.bm import logger
from buttermilk._core.agent import Agent  # Keep Agent import


# Define the Adapter Class
class AutogenAgentAdapter(RoutedAgent):
    """
    A transparent adapter that connects Buttermilk agents to the Autogen ecosystem.

    This adapter discovers methods decorated with @buttermilk_handler on the
    wrapped agent and registers them directly with Autogen's message routing system.
    """

    def __init__(self, agent_cfg: AgentConfig, wrapped_agent_cls: Type[Agent]):
        """
        Initialize the adapter with a Buttermilk agent.

        Args:
            agent_cfg: Configuration for the Buttermilk agent
            wrapped_agent_cls: Class of the Buttermilk agent to instantiate
        """
        # Initialize the parent RoutedAgent first
        # RoutedAgent only accepts description, not name
        super().__init__(description=agent_cfg.description)

        # Create the wrapped Buttermilk agent
        self.wrapped_agent = wrapped_agent_cls(**agent_cfg.model_dump())

        # Create message_handlers dictionary that will be populated with discovered handlers
        self.message_handlers: Dict[type, Callable] = {}

        # Flag to track if we've registered handlers
        self._handlers_initialized = True  # Set to true since we're registering handlers synchronously

        # Register handlers - we do this synchronously during init
        self._register_buttermilk_handlers()

    async def _register_buttermilk_handlers(self) -> None:
        """
        Discovers and registers handlers from the wrapped agent.
        This is called during initialization.
        """
        logger.debug(f"Adapter for '{self.wrapped_agent.name}': Registering handlers from buttermilk agent")

        # Iterate through members of the wrapped agent instance
        for name, method in inspect.getmembers(self.wrapped_agent):
            # Check if it's a method marked by our decorator
            if callable(method) and hasattr(method, "_buttermilk_handler_message_type"):
                target_message_type = getattr(method, "_buttermilk_handler_message_type", None)

                if target_message_type is None:
                    logger.warning(f"Method '{name}' on agent '{self.wrapped_agent.name}' is marked as a handler but missing message type. Skipping.")
                    continue

                logger.info(f"Adapter for '{self.wrapped_agent.name}': Found handler '{name}' for message type '{target_message_type.__name__}'")

                # Create a wrapper function that will be registered with Autogen.
                # This wrapper calls the original method on the wrapped_agent.
                async def handler_wrapper(
                    message: Any,  # Autogen message type determined by registration
                    ctx: MessageContext,
                    original_method: Callable = method,  # Capture method in closure
                    adapter_self=self,  # Capture adapter instance in closure
                ):
                    logger.debug(
                        f"Adapter for '{adapter_self.wrapped_agent.name}': Routing message type {type(message).__name__} to wrapped agent method '{original_method.__name__}'"
                    )
                    # Call the original method on the wrapped Buttermilk agent.
                    try:
                        # Inspect the method signature to determine how to pass parameters
                        sig = inspect.signature(original_method)
                        param_names = list(sig.parameters.keys())

                        if len(param_names) >= 1:
                            # Pass without naming the parameter (positional)
                            result = await original_method(message)
                        else:
                            # No parameters, just call the method
                            result = await original_method()

                        return result  # Return the result for Autogen's runtime
                    except Exception as e:
                        logger.error(
                            f"Error executing handler '{original_method.__name__}' on wrapped agent '{adapter_self.wrapped_agent.name}': {e}",
                            exc_info=True,
                        )
                        return None  # Default to None on error

                # Register the wrapper function with the specific message type
                if target_message_type:  # Ensure we have a valid type
                    try:
                        # Use the message_handler decorator to mark the handler
                        decorated_handler = message_handler(handler_wrapper)

                        # Set it as an attribute on the instance
                        handler_name = f"handle_{target_message_type.__name__}"
                        setattr(self, handler_name, decorated_handler)

                        # Store in our message handlers dictionary for direct access
                        self.message_handlers[target_message_type] = decorated_handler

                        logger.info(f"Adapter for '{self.wrapped_agent.name}': Registered handler '{name}' for type '{target_message_type.__name__}'")
                    except Exception as e:
                        logger.error(
                            f"Adapter for '{self.wrapped_agent.name}': Failed to register handler '{name}' for type '{target_message_type.__name__}': {e}",
                            exc_info=True,
                        )
                else:
                    logger.error(
                        f"Adapter for '{self.wrapped_agent.name}': Could not determine target message type for handler '{name}'. Skipping registration."
                    )

    # Delegate other important methods to the Buttermilk agent
    async def reset(self, cancellation_token=None):
        """Reset the agent's state."""
        if hasattr(self.wrapped_agent, "on_reset"):
            await self.wrapped_agent.on_reset(cancellation_token)

    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self.wrapped_agent, "cleanup"):
            await self.wrapped_agent.cleanup()

    async def initialize(self, **kwargs):
        """Initialize the agent."""
        if hasattr(self.wrapped_agent, "initialize"):
            await self.wrapped_agent.initialize(**kwargs)

    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> None:
        """Handle incoming group messages by delegating to the wrapped agent."""
        await self.wrapped_agent._listen(
            message=message,
            cancellation_token=ctx.cancellation_token,
            source=str(ctx.sender).split("/", maxsplit=1)[0] or "unknown",
        )


# --- Type checking block ---
if TYPE_CHECKING:
    from buttermilk._core.agent import Agent, AgentConfig
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
        ManagerResponse,
        OOBMessages,
        TaskProcessingComplete,
        TaskProcessingStarted,
        UserInstructions,
        AllMessages,
        ToolOutput,
    )
