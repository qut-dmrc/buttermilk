"""UI Proxy Agent implementation.

This module defines a UI Proxy Agent that dynamically connects to a concrete UI
implementation at runtime based on configuration or defaults.
"""

from collections.abc import Awaitable, Callable
from typing import Any, cast

from autogen_core import CancellationToken, MessageContext, message_handler
from pydantic import Field, PrivateAttr
from typing_extensions import Protocol

from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
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

    # Use Any for runtime flexibility, but we'll type check appropriately in our methods
    _concrete_ui: Any = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _ui_parameters: dict[str, Any] = PrivateAttr(default_factory=dict)

    async def initialize(self, session_id: str, callback_to_groupchat: Callable[..., Awaitable[None]], **kwargs) -> None:
        """Initialize the proxy agent and connect to a concrete UI implementation.
        
        Args:
            session_id: The session ID for this UI instance
            callback_to_groupchat: Callback function to send messages back to the group chat
            **kwargs: Additional parameters to pass to the concrete UI implementation

        """
        if not self.ui_type:
            self.ui_type = self.parameters.get("ui_type")
        # Copy these essential parameters to the UI parameters dictionary
        self._ui_parameters["session_id"] = self.session_id
        self._ui_parameters["callback_to_groupchat"] = callback_to_groupchat
        self._ui_parameters["name"] = self.ui_type
        await super().initialize(callback_to_groupchat=callback_to_groupchat, session_id=session_id, ui_type=self.ui_type, **kwargs)

        logger.info(f"UIProxyAgent initialized with: ui_type={self.ui_type}, session_id={self.session_id}, "
                   f"callback_to_groupchat={self._ui_parameters['callback_to_groupchat'] is not None}")

        # Connect to the concrete UI implementation if all required parameters are available
        if self.ui_type and self.session_id and self._ui_parameters["callback_to_groupchat"]:
            await self.connect_to_ui(self.ui_type)
        else:
            missing = []
            if not self.ui_type:
                missing.append("ui_type")
            if not self.session_id:
                missing.append("session_id")
            if not self._ui_parameters["callback_to_groupchat"]:
                missing.append("callback_to_groupchat")
            logger.warning(f"UIProxyAgent not fully initialized. Missing: {', '.join(missing)}")

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
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> AgentOutput:
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
            callback_to_groupchat = self._ui_parameters.get("callback_to_groupchat")
            logger.warning("UI proxy used before initialization or without concrete implementation")
            # If UI type is specified but not connected yet, try to connect
            if self.ui_type and self.session_id and callback_to_groupchat:
                await self.connect_to_ui(self.ui_type, callback_to_groupchat=callback_to_groupchat)
            else:
                available = ", ".join(list_ui_implementations())
                missing = []
                if not self.ui_type:
                    missing.append("ui_type")
                if not self.session_id:
                    missing.append("session_id")
                if not self._ui_parameters.get("callback_to_groupchat"):
                    missing.append("callback_to_groupchat")

                error_msg = f"No UI implementation connected. Missing parameters: {', '.join(missing)}. Available UIs: {available}"
                logger.error(error_msg)
                # Create a trace with error information
                return AgentTrace(
                    agent_id=self.agent_id,
                    session_id=self.session_id,
                    agent_info=self._cfg,
                    inputs=message,
                    outputs={"error": error_msg},
                    metadata={"error": True},
                )

        # Delegate to the concrete UI implementation
        if self._concrete_ui and hasattr(self._concrete_ui, "_process"):
            # Use dynamic call but ensure we return the right type
            result = await self._concrete_ui._process(
                inputs=message, cancellation_token=cancellation_token, **kwargs,
            )
            # The result should already be an AgentTrace or None, but cast to satisfy type checker
            if result is None:
                # Create a default trace for None results
                return AgentTrace(
                    agent_id=self.agent_id,
                    session_id=self.session_id,
                    agent_info=self._cfg,
                    inputs=message,
                    outputs={},
                    metadata={"no_response": True},
                )
            return result

        # Create a default trace if no concrete UI or process method
        return AgentTrace(
            agent_id=self.agent_id,
            session_id=self.session_id,
            agent_info=self._cfg,
            inputs=message,
            outputs={},
            metadata={"no_handler": True},
        )

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        source: str = "",
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
                message,
                cancellation_token=cancellation_token,
                public_callback=public_callback,
                message_callback=message_callback,
                source=source,
                **kwargs,
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
