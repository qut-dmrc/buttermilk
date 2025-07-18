"""Generic UI Agent base class.

This module defines a base UIAgent class that all concrete UI implementations should inherit from.
It establishes a consistent interface for UI agents in the system.
"""

import asyncio
from collections.abc import (
    Awaitable,
    Callable,
)
from typing import Any

from autogen_core import CancellationToken

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput, OOBMessages


class UIAgent(Agent):
    """Base class for all UI implementations.

    This class defines the common interface that all UI implementations must adhere to.
    It handles the basic initialization and resource management for UI agents.
    """

    def __init__(self, **kwargs):
        """Initialize UIAgent with callback configuration."""
        super().__init__(**kwargs)

        # Common fields that all UI implementations should have - moved from Field declaration
        self.callback_to_groupchat: Callable[..., Awaitable[None]] | None = kwargs.get("callback_to_groupchat")

        # Private attributes for internal state management - moved from PrivateAttr declaration
        self._input_task: asyncio.Task | None = None
        self._trace_this = False  # Controls whether this agent's messages are traced

    async def initialize(self, callback_to_groupchat: Callable[..., Awaitable[None]], **kwargs) -> None:
        """Initialize the UI agent with necessary callbacks and session info.

        Args:
            callback_to_groupchat: Callback function to send messages back to the group chat
            **kwargs: Additional parameters specific to the UI implementation

        """
        super().initialize(**kwargs)
        logger.debug(f"Initializing {self.__class__.__name__}")

        # Store the callback to groupchat for later use
        self.callback_to_groupchat = callback_to_groupchat

    # This is the abstract method that UI implementations must implement
    # The signature fully matches the base Agent class for compatibility
    async def _process(
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,
        **kwargs,
    ) -> AgentOutput:
        """Process inputs from the orchestrator and interact with the UI.

        Args:
            message: The input to process
            cancellation_token: Token for cancelling the operation
            public_callback: Callback for publishing messages to public topics
            **kwargs: Additional parameters

        Returns:
            The processing result

        This method must be implemented by concrete UI classes.

        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _process")

    # This matches the base Agent class's signature
    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,
        source: str = "",
        **kwargs,
    ) -> OOBMessages | None:
        """Process out-of-band control messages.

        Args:
            message: The control message to process
            cancellation_token: Token for cancelling the operation
            public_callback: Callback for publishing messages to public topics
            source: Identifier of the message sender
            **kwargs: Additional parameters

        Returns:
            Optional response to the control message

        UI agents typically need to listen to control messages to update the UI.

        """
        logger.debug(f"{self.__class__.__name__} received OOB message: {type(message).__name__}")
        return None

    # Default implementations for UI display - subclasses should override as needed
    async def callback_to_ui(self, message: Any, source: str = "") -> None:
        """Display a message to the UI. Override in subclasses for specific UI handling.

        Args:
            message: The message to display
            source: The source of the message

        """
        logger.debug(f"{self.__class__.__name__} received message from {source} for display")

    async def cleanup(self) -> None:
        """Clean up resources when the agent is no longer needed.

        This method should be implemented by concrete UI classes to properly
        release any resources like open connections, background tasks, etc.
        """
        logger.debug(f"Cleaning up {self.__class__.__name__}")

        # Cancel any input polling task if it exists
        if hasattr(self, "_input_task") and self._input_task and not self._input_task.done():
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                pass

    def make_callback(self) -> Callable[..., Awaitable[None]]:
        """Create a callback function for the UI agent.

        This method is a placeholder and should be implemented by concrete UI classes
        to provide the actual callback logic.

        Returns:
            A callable that can be used as a callback function

        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement make_callback")
