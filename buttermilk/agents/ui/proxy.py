"""UI Proxy Agent implementation.

This module defines a UI Proxy Agent that dynamically connects to a concrete UI
implementation at runtime based on configuration or defaults.
"""

from typing import Any, cast

from autogen_core import CancellationToken, MessageContext, message_handler
from pydantic import Field, PrivateAttr
from typing_extensions import Protocol

from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
    AllMessages,
    GroupchatMessageTypes,
    OOBMessages,
)
from buttermilk.agents.ui.generic import UIAgent
from buttermilk.agents.ui.registry import get_ui_implementation, list_ui_implementations


# Define a protocol for methods we expect on a UI implementation
class UIImplementation(Protocol):
    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentTrace | None: ...
    async def _handle_events(self, message: OOBMessages, cancellation_token: CancellationToken | None = None, **kwargs) -> OOBMessages | None: ...
    async def handle_groupchat_message(self, message: GroupchatMessageTypes, ctx: MessageContext) -> None: ...
    async def handle_control_message(self, message: AllMessages, ctx: MessageContext) -> OOBMessages | None: ...
    async def cleanup(self) -> None: ...


class UIProxyAgent(UIAgent):
    """A proxy agent that delegates to a concrete UI implementation.
    
    This agent serves as an intermediary between the flow orchestrator and the actual UI
    implementation. It dynamically connects to a specific UI implementation at runtime
    based on configuration.
    """

    ui_type: str | None = Field(default=None, description="The type of UI to use")
    session_id: str = Field(description="Session identifier for the UI")
    # Use Any for runtime flexibility, but we'll type check appropriately in our methods
    _concrete_ui: Any = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _ui_parameters: dict[str, Any] = PrivateAttr(default_factory=dict)

    async def initialize(self, **kwargs) -> None:
        """Initialize the proxy agent and connect to a concrete UI implementation.
        
        Args:
            **kwargs: Additional parameters to pass to the concrete UI implementation

        """
        await super().initialize(**kwargs)

        # Store parameters for later use with the concrete UI
        self._ui_parameters = kwargs
        self._ui_parameters["callback_to_ui"] = self.parameters.get("callback_to_ui", self.callback_to_ui)
        self._ui_parameters["session_id"] = self.parameters.get("session_id", self.session_id)
        self._ui_parameters["name"] = self.ui_type
        self.ui_type = self.parameters.get("ui_type", self.ui_type)

        # Connect to the concrete UI implementation if available
        if self.ui_type:
            await self.connect_to_ui(self.ui_type)

    async def connect_to_ui(self, ui_type: str) -> None:
        """Connect to a specific UI implementation.
        
        Args:
            ui_type: The type of UI to connect to
            
        Raises:
            ValueError: If the UI implementation is not registered

        """
        try:
            # Get the UI implementation class from the registry
            ui_class = get_ui_implementation(ui_type)

            # Create an instance of the UI implementation
            self._concrete_ui = ui_class(
                description=f"Concrete UI implementation ({ui_type})",
                **self._ui_parameters,
            )

            # Initialize the concrete UI if it has an initialize method
            if hasattr(self._concrete_ui, "initialize"):
                await self._concrete_ui.initialize(**self._ui_parameters)

            self.ui_type = ui_type
            self._initialized = True
            logger.info(f"Connected to UI implementation: {ui_type}")
        except Exception as e:
            available = ", ".join(list_ui_implementations())
            logger.error(f"Failed to connect to UI '{ui_type}': {e}. Available UIs: {available}")
            raise

    async def _process(
        self, *, inputs: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs,
    ) -> AgentTrace | None:
        """Process inputs by delegating to the concrete UI implementation.
        
        Args:
            inputs: The input to process
            cancellation_token: Token for cancelling the operation
            **kwargs: Additional parameters
            
        Returns:
            The processing result from the concrete UI
            
        Raises:
            ValueError: If no concrete UI implementation is connected

        """
        if not self._initialized or not self._concrete_ui:
            logger.warning("UI proxy used before initialization or without concrete implementation")
            # If UI type is specified but not connected yet, try to connect
            if self.ui_type:
                await self.connect_to_ui(self.ui_type)
            else:
                available = ", ".join(list_ui_implementations())
                error_msg = f"No UI implementation connected. Available: {available}"
                logger.error(error_msg)
                # We cannot create a proper AgentTrace without required parameters,
                # so return None with logging instead
                return None

        # Delegate to the concrete UI implementation
        if self._concrete_ui and hasattr(self._concrete_ui, "_process"):
            # Use dynamic call but ensure we return the right type
            result = await self._concrete_ui._process(
                inputs=inputs, cancellation_token=cancellation_token, **kwargs,
            )
            # The result should already be an AgentTrace or None, but cast to satisfy type checker
            return cast("AgentTrace | None", result)

        return None

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> OOBMessages | None:
        """Handle events by delegating to the concrete UI implementation.
        
        Args:
            message: The message to handle
            cancellation_token: Token for cancelling the operation
            **kwargs: Additional parameters
            
        Returns:
            The result from the concrete UI

        """
        if not self._initialized or not self._concrete_ui:
            logger.warning("UI proxy used before initialization or without concrete implementation")
            return None

        # Delegate to the concrete UI implementation
        if self._concrete_ui and hasattr(self._concrete_ui, "_handle_events"):
            return await self._concrete_ui._handle_events(
                message, cancellation_token=cancellation_token, **kwargs,
            )

        return None

    @message_handler
    async def handle_groupchat_message(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext,
    ) -> None:
        """Handle messages from the group chat by forwarding to the concrete UI.
        
        Args:
            message: The group chat message
            ctx: Message context

        """
        if not self._initialized or not self._concrete_ui:
            logger.warning("UI proxy received message before initialization")
            return

        # Delegate to the concrete UI if it has a handle_groupchat_message method
        if self._concrete_ui and hasattr(self._concrete_ui, "handle_groupchat_message"):
            # Use dynamic call to bypass static type checking
            await self._concrete_ui.handle_groupchat_message(message, ctx)

    @message_handler
    async def handle_control_message(
        self,
        message: AllMessages,
        ctx: MessageContext,
    ) -> OOBMessages | None:
        """Handle control messages by delegating to the concrete UI.
        
        Args:
            message: The control message
            ctx: Message context
            
        Returns:
            The result from the concrete UI

        """
        if not self._initialized or not self._concrete_ui:
            logger.warning("UI proxy received control message before initialization")
            return None

        # Delegate to the concrete UI if it has a handle_control_message method
        if self._concrete_ui and hasattr(self._concrete_ui, "handle_control_message"):
            # Use dynamic call to bypass static type checking
            result = await self._concrete_ui.handle_control_message(message, ctx)
            return cast("OOBMessages | None", result)

        return None

    async def cleanup(self) -> None:
        """Clean up resources by delegating to the concrete UI."""
        if self._initialized and self._concrete_ui and hasattr(self._concrete_ui, "cleanup"):
            # Use dynamic call to bypass static type checking
            await self._concrete_ui.cleanup()

        self._initialized = False
        self._concrete_ui = None
