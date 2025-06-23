"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from pydantic import PrivateAttr

from buttermilk._core import AgentInput, StepRequest, logger
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentAnnouncement, AgentOutput, AgentTrace, GroupchatMessageTypes
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent
from buttermilk.utils._tools import create_tool_functions


class StructuredLLMHostAgent(LLMAgent, HostAgent):
    """Host agent that uses structured tool definitions for agent coordination.

    This agent replaces natural language agent descriptions with structured
    tool definitions, enabling more reliable and type-safe agent invocation.
    """

    _user_feedback: list[str] = PrivateAttr(default_factory=list)
    _proposed_step: asyncio.Queue[StepRequest] = PrivateAttr(default_factory=asyncio.Queue)

    # Override the output model - we don't need CallOnAgent anymore
    _output_model = None  # Let the LLM use tool calling directly

    def _clear_pending_steps(self) -> None:
        """Clear all pending steps from the queue."""
        while not self._proposed_step.empty():
            try:
                self._proposed_step.get_nowait()
                logger.debug("Cleared pending step from queue due to new manager request")
            except asyncio.QueueEmpty:
                break

    def set_participant_tools_for_testing(self, participant_tools: dict[str, Any]) -> None:
        """Set participant tools for testing purposes.
        
        This method allows tests to populate the _participant_tools directly
        without going through the full orchestrator initialization.
        """
        self._participant_tools = participant_tools

    async def initialize(self, callback_to_groupchat: Any, **kwargs: Any) -> None:
        """Initialize the host agent."""
        await super().initialize(callback_to_groupchat=callback_to_groupchat, **kwargs)
        await self._initialize(callback_to_groupchat=callback_to_groupchat)

    async def _initialize(self, callback_to_groupchat: Any) -> None:
        """Initialize the host with agent-provided tool definitions."""
        # Note: super().initialize() is already called in our initialize() method

        # Build initial tools (may be empty if no agents announced yet)
        await self._build_agent_tools()

        # Tools will be rebuilt automatically as agents announce themselves
        # No need to wait for participants list - we use the agent registry instead
        logger.debug(f"StructuredLLMHost {self.agent_name} initialized with {len(self._tools_list)} tools. More tools will be built from agent announcements.")

    async def _build_agent_tools(self) -> None:
        """Build tools from agent-provided tool definitions.
        
        This method extracts actual @tool decorated methods from agents and wraps them
        to send StepRequest messages, following the agent-centric approach.
        """
        agent_tools = []

        # Build tools from agent announcements 
        for agent_id, announcement in self._agent_registry.items():
            agent_role = announcement.agent_config.role
            
            # Get the actual agent instance to extract its tools
            # We need to get the agent instance to access its @tool decorated methods
            agent_instance = None
            # TODO: We need a way to get the actual agent instance from the announcement
            # For now, use the tool_definition as a fallback
            
            if announcement.tool_definition:
                tool_def = announcement.tool_definition
                
                # Create wrapper function that sends StepRequest
                def create_tool_wrapper(target_role: str, tool_name: str, tool_description: str):
                    # Create function that matches the expected signature
                    async def tool_wrapper(**kwargs: Any) -> None:
                        """Wrapper that sends StepRequest to agent."""
                        step_request = StepRequest(
                            role=target_role,
                            inputs=kwargs,
                            metadata={"tool_name": tool_name}
                        )
                        logger.info(f"Host {self.agent_name} invoking {target_role} tool {tool_name}")
                        await self._proposed_step.put(step_request)
                    
                    return tool_wrapper

                # Create FunctionTool that wraps the agent call
                tool = FunctionTool(
                    func=create_tool_wrapper(agent_role, tool_def["name"], tool_def.get("description", "")),
                    name=tool_def["name"],
                    description=tool_def.get("description", f"Call {agent_role} agent")
                )
                agent_tools.append(tool)
                logger.debug(f"Registered tool wrapper: {tool_def['name']} for {agent_role}")

        # Initialize tools list using parent class pattern
        if self.tools:
            configured_tools = create_tool_functions(self.tools)
            self._tools_list = configured_tools.copy()
        else:
            self._tools_list = []

        # Add all agent tools, avoiding duplicates
        for tool in agent_tools:
            if not any(existing.name == tool.name for existing in self._tools_list):
                self._tools_list.append(tool)

        tool_names = [tool.name for tool in self._tools_list]
        logger.info(
            f"Structured LLMHost built {len(agent_tools)} agent tools "
            f"from {len(self._agent_registry)} announced agents, "
            f"total tools: {len(self._tools_list)} - {tool_names}"
        )

    async def update_agent_registry(self, announcement: AgentAnnouncement) -> None:
        """Update registry and rebuild tools when agents announce."""
        # Call parent class to handle the registry update
        await super().update_agent_registry(announcement)
        # Rebuild tools whenever registry changes
        await self._build_agent_tools()

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute."""
        # First, say hello to the user
        await asyncio.sleep(3)  # Let the group chat initialize
        yield StepRequest(
            role=MANAGER,
            content="Hi! What would you like to do?",
        )

        while True:
            # Wait for the _listen method to add a proposed step to the queue
            task = await self._proposed_step.get()
            yield task

            # Check if this is an END task to break the loop
            if task.role == END:
                break

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken,
        source: str = "",
        public_callback: Callable,
        message_callback: Callable,
        **kwargs: Any,
    ) -> None:
        """Listen to messages and use structured tools to determine next steps."""
        # Save messages to context
        await super()._listen(
            message=message,
            cancellation_token=cancellation_token,
            source=source,
            public_callback=public_callback,
            message_callback=message_callback,
            **kwargs,
        )

        if isinstance(message, ManagerMessage):
            # Skip command messages
            if message.content and str(message.content).startswith(COMMAND_SYMBOL):
                return

            # Skip empty messages
            if not message.content:
                logger.debug("Manager message received with empty content, skipping")
                return

            # Clear any pending steps since the manager has a new request
            self._clear_pending_steps()

            logger.info(f"Manager interrupted with new request: {message.content}")

            # Use the LLM with structured tools to determine next step
            # The template should be configured to use tool calling
            result = await self.invoke(
                message=AgentInput(
                    inputs={
                        "user_feedback": self._user_feedback,
                        "prompt": str(message.content),
                        "available_agents": [ann.agent_config.role for ann in self._agent_registry.values()]
                    }
                ),
                public_callback=public_callback,
                message_callback=message_callback,
                cancellation_token=cancellation_token,
                **kwargs
            )

            if result:
                # The LLM may have called a tool, which will have already
                # sent the appropriate StepRequest via the tool's callback
                logger.debug(
                    "LLM response processed. Tool calls handled automatically via FunctionTool callbacks."
                )

                # Just pass the response to the manager
                await self.callback_to_groupchat(result)
