"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel, PrivateAttr

from buttermilk._core import AgentInput, StepRequest, logger
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentAnnouncement, AgentOutput, AgentTrace, GroupchatMessageTypes
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent
from buttermilk.utils._tools import create_tool_functions


class AgentProxyTool(Tool):
    """Simple tool that proxies calls to agents via StepRequest."""
    
    def __init__(
        self, 
        role: str,
        tool_schema: ToolSchema,
        proposed_step_queue: asyncio.Queue[StepRequest],
        host_agent_name: str
    ):
        # Initialize base Tool
        super().__init__(
            args_type=BaseModel,
            return_type=dict
        )
        self.role = role
        self.tool_schema = tool_schema
        self.proposed_step_queue = proposed_step_queue
        self.host_agent_name = host_agent_name
    
    @property
    def name(self) -> str:
        return self.tool_schema["name"]
    
    @property
    def description(self) -> str:
        return self.tool_schema.get("description", f"Call {self.role} agent")
    
    @property
    def schema(self) -> ToolSchema:
        """Return the autogen ToolSchema directly."""
        return self.tool_schema
    
    async def run_json(self, args: str, cancellation_token: CancellationToken) -> str:
        """Execute the tool by sending a StepRequest to the target agent."""
        # Parse the JSON args
        inputs = json.loads(args) if args else {}
        
        step_request = StepRequest(
            role=self.role,
            inputs=inputs,
            metadata={"tool_name": self.name}
        )
        logger.info(f"Host {self.host_agent_name} invoking {self.role} tool {self.name}")
        await self.proposed_step_queue.put(step_request)
        
        return json.dumps({"status": "queued", "agent": self.role})


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

    async def initialize(self, callback_to_groupchat: Any, **kwargs: Any) -> None:
        """Initialize the host agent."""
        await super().initialize(callback_to_groupchat=callback_to_groupchat, **kwargs)

        # Build initial tools (may be empty if no agents announced yet)
        await self._build_agent_tools()

        # Tools will be rebuilt automatically as agents announce themselves
        # No need to wait for participants list - we use the agent registry instead
        logger.debug(f"StructuredLLMHost {self.agent_name} initialized with {len(self._tools_list)} tools. More tools will be built from agent announcements.")

    async def _build_agent_tools(self) -> None:
        """Build tools from agent-provided tool definitions.
        
        This method uses the tool definitions already created by agents and wraps them
        with a lightweight proxy to send StepRequest messages through the groupchat.
        """
        agent_tools = []

        # Build tools from agent announcements
        for agent_id, announcement in self._agent_registry.items():
            agent_role = announcement.agent_config.role

            if tool_def := announcement.tool_definition:
                # Create a simple proxy tool that uses the agent's schema directly
                tool = AgentProxyTool(
                    role=agent_role,
                    tool_schema=tool_def,
                    proposed_step_queue=self._proposed_step,
                    host_agent_name=self.agent_name
                )

                agent_tools.append(tool)
                logger.debug(f"Registered tool: {tool_def['name']} for {agent_role}")

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
        # First, say hello to the user, and send a message to all other participants
        # to trigger them to announce.
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
        **kwargs: Any,
    ) -> None:
        """Listen to messages and use structured tools to determine next steps."""
        # Save messages to context
        await super()._listen(
            message=message,
            cancellation_token=cancellation_token,
            source=source,
            public_callback=public_callback,
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
