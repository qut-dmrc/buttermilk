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
    Adapter to wrap a Buttermilk Agent (like LLMAgent/Judge) for use with Autogen.

    It dynamically discovers methods marked with @buttermilk_handler on the wrapped
    agent and registers corresponding handlers with the Autogen runtime.
    """

    def __init__(self, agent_cfg: AgentConfig, wrapped_agent_cls: Type[Agent]):
        """
        Initializes the adapter.

        Args:
            agent_cfg: The Buttermilk Agent configuration to use.
            wrapped_agent_cls: The Buttermilk Agent class to instantiate.
        """
        # Instantiate the wrapped agent using the provided class and configuration
        self.wrapped_agent = wrapped_agent_cls(**agent_cfg.model_dump())

        # Initialize the parent RoutedAgent.
        super().__init__(description=self.wrapped_agent.description)  # Pass description from wrapped agent

        # Create message_handlers dictionary that will be populated on demand
        self.message_handlers = {}

        # Flag to track if we've scanned for handlers
        self._handlers_initialized = False

    async def _register_buttermilk_handlers(self) -> None:
        """
        Discovers and registers handlers from the wrapped agent.
        This is called lazily when the first message is received.
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
                        # Use the RoutedAgent's method to register handlers
                        self.message_handlers[target_message_type] = handler_wrapper
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

    # Implement message handling method to intercept messages and lazy-register handlers
    async def handle_message(self, message: Any, ctx: MessageContext) -> Any:
        """
        Intercept incoming messages to ensure handlers are registered before processing.
        This allows handlers to be registered on-demand when the first message arrives.
        """
        # If handlers are not initialized yet, register them
        if not self._handlers_initialized:
            logger.info(f"Adapter for '{self.wrapped_agent.name}': Lazily registering handlers on first message")
            await self._register_buttermilk_handlers()
            self._handlers_initialized = True

        # Delegate to parent class handler
        return await super().handle_message(message, ctx)

    # Delegate necessary Agent lifecycle methods
    async def reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Delegates reset to the wrapped agent if it has an on_reset method."""
        logger.debug(f"Adapter for '{self.wrapped_agent.name}': Delegating reset()")
        if hasattr(self.wrapped_agent, "on_reset") and callable(self.wrapped_agent.on_reset):
            await self.wrapped_agent.on_reset(cancellation_token=cancellation_token)
        # Call super().reset() if RoutedAgent has its own reset logic to perform
        await super().reset(cancellation_token=cancellation_token)

    async def cleanup(self) -> None:
        """Delegates cleanup to the wrapped agent if it has a cleanup method."""
        logger.debug(f"Adapter for '{self.wrapped_agent.name}': Delegating cleanup()")
        if hasattr(self.wrapped_agent, "cleanup") and callable(self.wrapped_agent.cleanup):
            await self.wrapped_agent.cleanup()
        # Call super().cleanup() if RoutedAgent has its own cleanup logic
        if hasattr(super(), "cleanup") and callable(super().cleanup):
            await super().cleanup()

    # Add initialization to ensure the wrapped agent is properly set up
    async def initialize(self, **kwargs) -> None:
        """Initialize the wrapped agent if it has an initialize method."""
        logger.debug(f"Adapter for '{self.wrapped_agent.name}': Initializing wrapped agent")
        if hasattr(self.wrapped_agent, "initialize") and callable(self.wrapped_agent.initialize):
            await self.wrapped_agent.initialize(**kwargs)

        # Call super().initialize() if RoutedAgent has initialization logic
        if hasattr(super(), "initialize") and callable(super().initialize):
            await super().initialize(**kwargs)

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
