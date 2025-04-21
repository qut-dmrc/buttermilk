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

    def __init__(self, agent_cfg: AgentConfig):
        """
        Initializes the adapter.

        Args:
            agent: The Buttermilk Agent instance to wrap.
        """
        if not isinstance(agent_cfg, Agent):
            raise TypeError(f"Expected a Buttermilk Agent instance, but got {type(agent_cfg)}")

        self.wrapped_agent = agent_cfg

        # Initialize the parent BaseAgent (via RoutedAgent).
        # BaseAgent.__init__ accepts name and description.
        super().__init__(
            name=self.wrapped_agent.name,  # Pass name from wrapped agent
            description=self.wrapped_agent.description,  # Pass description from wrapped agent
            # Pass other RoutedAgent args if needed, e.g., system_message
            # system_message=getattr(self.wrapped_agent, 'system_message', None)
        )

        # --- Dynamic Handler Registration ---
        self._register_buttermilk_handlers()

    def _register_buttermilk_handlers(self):
        """
        Inspects the wrapped agent for methods decorated with @buttermilk_handler
        and registers them with the Autogen runtime.
        """
        logger.debug(f"Adapter for '{self.name}': Searching for @buttermilk_handlers on {type(self.wrapped_agent).__name__}")

        # Iterate through members of the wrapped agent instance
        for name, member in inspect.getmembers(self.wrapped_agent):
            # Check if it's a method marked by our decorator
            if callable(member) and hasattr(member, "_is_buttermilk_handler") and member._is_buttermilk_handler:
                target_message_type: Type = getattr(member, "_buttermilk_handler_message_type", None)

                if target_message_type is None:
                    logger.warning(f"Method '{name}' on agent '{self.wrapped_agent.name}' is marked as a handler but missing message type. Skipping.")
                    continue

                logger.info(f"Adapter for '{self.name}': Found handler '{name}' for message type '{target_message_type.__name__}'")

                # Create a wrapper function that will be registered with Autogen.
                # This wrapper calls the original method on the wrapped_agent.
                async def handler_wrapper(
                    message: Any,  # Autogen message type determined by registration
                    ctx: MessageContext,
                    original_method: Callable = member,  # Capture method in closure
                    adapter_self=self,  # Capture adapter instance in closure
                ):
                    # The 'name' attribute comes from BaseAgent initialization
                    logger.debug(
                        f"Adapter '{adapter_self.name}': Routing message type {type(message).__name__} to wrapped agent method '{original_method.__name__}'"
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
                        # register_handler should be inherited from BaseAgent
                        self.register_handler(target_message_type, handler_wrapper)
                        logger.info(f"Adapter for '{self.name}': Registered handler '{name}' for type '{target_message_type.__name__}'")
                    except Exception as e:
                        logger.error(
                            f"Adapter for '{self.name}': Failed to register handler '{name}' for type '{target_message_type.__name__}': {e}",
                            exc_info=True,
                        )
                else:
                    logger.error(f"Adapter for '{self.name}': Could not determine target message type for handler '{name}'. Skipping registration.")

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
