from __future__ import annotations  # Add this import at the very top

import asyncio
import inspect  # Import inspect for introspection
from functools import partial  # Import partial for creating wrapper methods
from collections.abc import Awaitable, Callable
from typing import Any, Optional, Union, Sequence, TYPE_CHECKING, List, Dict, Callable, Awaitable, Type  # Added Type

from autogen_core import (
    DefaultTopicId,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId,
    # message_handler is not used directly here, registration is manual
)

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
        # Note: RoutedAgent takes different parameters than we thought
        # We'll just use the description as the primary identifier
        super().__init__(description=self.wrapped_agent.description)  # Pass description from wrapped agent
        
        # Create message_handlers dictionary if it doesn't exist
        if not hasattr(self, "message_handlers"):
            self.message_handlers = {}
        
        # --- Dynamic Handler Registration ---
        self._register_buttermilk_handlers()

    def _register_buttermilk_handlers(self):
        """
        Inspects the wrapped agent for methods decorated with @buttermilk_handler
        and registers them with the Autogen runtime.
        """
        logger.debug(f"Adapter for '{self.wrapped_agent.name}': Searching for @buttermilk_handlers on {type(self.wrapped_agent).__name__}")

        # Iterate through members of the wrapped agent instance
        for name, member in inspect.getmembers(self.wrapped_agent):
            # Check if it's a method marked by our decorator
            if callable(member) and hasattr(member, "_buttermilk_handler_message_type"):
                target_message_type = getattr(member, "_buttermilk_handler_message_type", None)

                if target_message_type is None:
                    logger.warning(f"Method '{name}' on agent '{self.wrapped_agent.name}' is marked as a handler but missing message type. Skipping.")
                    continue

                logger.info(f"Adapter for '{self.wrapped_agent.name}': Found handler '{name}' for message type '{target_message_type.__name__}'")

                # Create a wrapper function that will be registered with Autogen.
                # This wrapper calls the original method on the wrapped_agent.
                async def handler_wrapper(
                    message: Any,  # Autogen message type determined by registration
                    ctx: MessageContext,
                    original_method: Callable = member,  # Capture method in closure
                    adapter_self=self,  # Capture adapter instance in closure
                ):
                    logger.debug(
                        f"Adapter for '{adapter_self.wrapped_agent.name}': Routing message type {type(message).__name__} to wrapped agent method '{original_method.__name__}'"
                    )
                    # Call the original method on the wrapped Buttermilk agent.
                    # Pass only the message, as the decorated method might not expect 'ctx'.
                    # The original method should return values appropriate for its context
                    # (e.g., AgentOutput for processing, maybe None for listening).
                    # The adapter doesn't typically modify or publish the return value here,
                    # unless a specific adaptation is needed. Autogen runtime handles publishing based on registration.
                    try:
                        # We assume the decorated method takes 'message' as the primary argument.
                        # If it needs more adaptation (e.g. extracting from ctx), the wrapper needs modification.
                        result = await original_method(message=message)  # Pass message directly
                        return result  # Return the result for Autogen's runtime
                    except Exception as e:
                        logger.error(
                            f"Error executing handler '{original_method.__name__}' on wrapped agent '{adapter_self.wrapped_agent.name}': {e}",
                            exc_info=True,
                        )
                        # Optionally, return an error response or raise? Depends on Autogen's error handling.
                        return None  # Default to None on error

                # Register the wrapper function with the specific message type
                # The type hint on the 'message' parameter of the wrapper doesn't
                # strictly enforce the type for registration, but Autogen uses the
                # message_type argument here.
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

    # --- Optional: Delegate other necessary Agent methods? ---
    # If the Autogen runtime directly calls methods other than registered handlers
    # (e.g., maybe 'reset' or specific lifecycle methods), you might need to
    # explicitly delegate them here. Check RoutedAgent/BaseAgent definition if needed.

    # Example delegation (uncomment and adapt if needed):
    # async def reset(self, cancellation_token: CancellationToken | None = None) -> None:
    #     """Delegates reset to the wrapped agent if it has an on_reset method."""
    #     logger.debug(f"Adapter for '{self.name}': Delegating reset()")
    #     if hasattr(self.wrapped_agent, 'on_reset') and callable(self.wrapped_agent.on_reset):
    #          await self.wrapped_agent.on_reset(cancellation_token=cancellation_token)
    #     # Call super().reset() if RoutedAgent has its own reset logic to perform
    #     await super().reset(cancellation_token=cancellation_token)

    # Ensure the adapter uses the wrapped agent's core processing if needed,
    # although typically Autogen relies on registered handlers.
    # The __call__ method might not be directly used by Autogen's runtime
    # in the same way it is in the Buttermilk orchestrator.


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
