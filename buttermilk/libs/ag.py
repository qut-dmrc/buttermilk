# Example: Create a new file, maybe buttermilk/_integration/autogen_adapter.py
import inspect
from typing import Annotated, Any, Dict, Optional, Union, Type
from autogen_core import Agent as AutogenAgent, RoutedAgent, MessageContext, CancellationToken, message_handler

from buttermilk._core.config import AgentConfig as ButtermilkAgentConfig  # Your base Buttermilk agent
from buttermilk._core.contract import AgentInput  # Example message type
from buttermilk._core.log import logger


class AutogenAgentAdapter(RoutedAgent):
    """
    Wraps a Buttermilk Agent instance to allow it to participate
    in an Autogen group chat using RoutedAgent capabilities.
    """

    def __init__(self, agent_cfg: ButtermilkAgentConfig, wrapped_agent_cls, **routed_agent_kwargs: Any):
        """
        Args:
            buttermilk_agent: The instance of the Buttermilk agent to wrap.
            **routed_agent_kwargs: Keyword arguments to pass to the RoutedAgent constructor
                                    (e.g., name, description, system_message).
        """
        # Ensure essential RoutedAgent args are provided
        if "description" not in routed_agent_kwargs:
            routed_agent_kwargs["description"] = agent_cfg.description

        self._buttermilk_agent = wrapped_agent_cls(**agent_cfg.model_dump())

        self._register_buttermilk_handlers()

        super().__init__(**routed_agent_kwargs)

    def _register_buttermilk_handlers(self):
        """
        Inspects the wrapped Buttermilk agent for methods decorated with
        @buttermilk_handler and registers them with Autogen's handler mechanism.
        """
        for _, method in inspect.getmembers(self._buttermilk_agent, predicate=inspect.ismethod):
            if hasattr(method, "_buttermilk_handler_message_type"):
                message_types: Type = getattr(method, "_buttermilk_handler_message_type")
                handler_func = self._create_autogen_handler(method, message_types=message_types)
                # Register the handler on THIS adapter instance for the specific type
                setattr(self, method.__name__, message_handler(handler_func))
                logger.debug(f"Registered handler for {[message_types]} on -> {method.__name__}")

    def _create_autogen_handler(self, buttermilk_method, message_types):
        """
        Creates a wrapper function suitable for Autogen's register_handler,
        which calls the original Buttermilk method.
        """

        async def autogen_handler_wrapper(
            message: message_types,  # The specific message type Autogen provides
            ctx: MessageContext,
            # cancellation_token: Optional[CancellationToken] = None # Add if needed
        ) -> Optional[Any]:  # Define expected return type for Autogen
            """Handles message dispatch from Autogen to the Buttermilk agent."""
            print(f"Adapter {self.__qualname__} routing message of type {type(message)} to {buttermilk_method.__name__}")  # Debug
            # Here, you might need translation if the buttermilk method expects
            # a different signature or context than Autogen provides.
            # For simplicity, assuming the decorated method can handle the message directly:
            try:
                result = await buttermilk_method(message)

                if result is not None:
                    await self.runtime.publish_message(
                        message=result,  # Autogen expects dict/str usually
                        topic_id=ctx.topic_id,  # Use the context's topic
                        sender=self.id,  # Send as the adapter agent
                    )
                    return None  # Indicate message handled, result published
                else:
                    # Buttermilk method returned None (maybe it published itself, or did nothing)
                    return None  # Indicate message handled

            except Exception as e:
                print(f"Error executing Buttermilk handler {buttermilk_method.__name__}: {e}")
                # Handle error appropriately - maybe log, publish an error message, or re-raise
                error_message = {"error": f"Handler failed: {e}"}
                await self.runtime.publish_message(message=error_message, topic_id=ctx.topic_id, sender=self.id)
                return None  # Indicate message handled (by erroring)

        return autogen_handler_wrapper
