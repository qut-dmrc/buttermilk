"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from pydantic import Field, PrivateAttr

from buttermilk._core import AgentInput, StepRequest, logger
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentTrace, ConductorRequest, GroupchatMessageTypes, OOBMessages
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent


class StructuredLLMHostAgent(LLMAgent, HostAgent):
    """Host agent that uses structured tool definitions for agent coordination.
    
    This agent replaces natural language agent descriptions with structured
    tool definitions, enabling more reliable and type-safe agent invocation.
    """

    _user_feedback: list[str] = PrivateAttr(default_factory=list)
    _proposed_step: asyncio.Queue[StepRequest] = PrivateAttr(default_factory=asyncio.Queue)
    max_user_confirmation_time: int = Field(
        default=7200,
        description="Maximum time to wait for agent responses in seconds",
    )

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
        await self._initialize(callback_to_groupchat=callback_to_groupchat)

    async def _initialize(self, callback_to_groupchat: Any) -> None:
        """Initialize the host with structured tool definitions from participants."""
        # Note: super().initialize() is already called in our initialize() method

        # Check if participants are available yet
        if not self._participants:
            logger.warning(f"No participants available during {self.agent_name} initialization. Tools will be set up when participants are available.")
            return

        # Build agent tools from participants
        await self._build_agent_tools()

    async def _build_agent_tools(self) -> None:
        """Build structured tool definitions from participants.
        
        This method creates FunctionTool objects for each participant's capabilities.
        Each tool generates a StepRequest message addressed to STEP (not individual agents).
        The tools are stored in _tools_list to be used by the parent LLMAgent class.
        """
        agent_tools = []
        tool_names_seen = set()  # Track tool names to avoid duplicates

        for role, description in self._participants.items():
            # Check if we have specific tool definitions for this role
            if hasattr(self, '_participant_tools') and role in self._participant_tools:
                logger.debug(f"Using provided tool definitions for role {role}")
                tool_defs = self._participant_tools[role]
            else:
                # Create a default tool for this role
                logger.debug(f"Creating default tool for role {role}")
                tool_defs = [{
                    'name': f"call_{role.lower()}",
                    'description': f"Send a request to the {role} agent: {description}",
                    'input_schema': {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": f"The request or question for the {role} agent"
                            }
                        },
                        "required": ["prompt"]
                    },
                    'output_schema': {"type": "object"}
                }]

            # Create FunctionTool for each tool definition
            for tool_dict in tool_defs:
                tool_name = tool_dict['name']
                # Create unique tool name to avoid duplicates
                unique_tool_name = f"{role.lower()}.{tool_name}"

                # Skip if we've already seen this exact tool name
                if unique_tool_name in tool_names_seen:
                    logger.debug(f"Skipping duplicate tool: {unique_tool_name}")
                    continue

                tool_names_seen.add(unique_tool_name)

                # Create a closure to capture the role and tool name
                def create_agent_tool(agent_role: str, tool_id: str, tool_schema: dict) -> Callable:
                    """Create a tool function that generates StepRequest messages."""
                    # Extract parameters from schema
                    properties = tool_schema.get("properties", {})

                    # Build a function with explicit parameters based on schema
                    # For simplicity, we'll handle common cases
                    if len(properties) == 1 and "prompt" in properties:
                        # Simple case: just a prompt parameter
                        async def call_agent_with_prompt(prompt: str) -> dict[str, Any]:
                            """Call a specific tool on an agent."""
                            # Create StepRequest addressed to STEP, not the individual agent
                            step_request = StepRequest(
                                role=agent_role,  # The agent role to invoke
                                inputs={
                                    "tool": tool_id,
                                    "tool_inputs": {"prompt": prompt}
                                }
                            )
                            logger.info(
                                f"Host {self.agent_name} invoking {agent_role}.{tool_id} "
                                f"with prompt: {prompt[:100]}..."
                            )
                            # Put the step request in the queue for _sequence to yield
                            await self._proposed_step.put(step_request)
                            return {"status": "queued", "step": agent_role}
                        return call_agent_with_prompt
                    else:
                        # Generic case: accept a dict of inputs
                        async def call_agent_with_inputs(**kwargs: Any) -> dict[str, Any]:
                            """Call a specific tool on an agent with arbitrary inputs."""
                            # Create StepRequest addressed to STEP
                            step_request = StepRequest(
                                role=agent_role,
                                inputs={
                                    "tool": tool_id,
                                    "tool_inputs": kwargs
                                }
                            )
                            logger.info(
                                f"Host {self.agent_name} invoking {agent_role}.{tool_id} "
                                f"with inputs: {list(kwargs.keys())}"
                            )
                            # Put the step request in the queue
                            await self._proposed_step.put(step_request)
                            return {"status": "queued", "step": agent_role}
                        return call_agent_with_inputs

                # Create the actual tool function
                tool_func = create_agent_tool(role, tool_name, tool_dict.get('input_schema', {}))

                # Create FunctionTool with proper schema
                function_tool = FunctionTool(
                    func=tool_func,
                    name=unique_tool_name,
                    description=tool_dict['description']
                )

                agent_tools.append(function_tool)
                logger.debug(f"Registered tool: {unique_tool_name}")

        # Initialize tools list using parent class pattern
        # First, load any configured tools from the agent config
        if self.tools:
            # This uses the parent class's tool loading mechanism
            from buttermilk.utils._tools import create_tool_functions
            configured_tools = create_tool_functions(self.tools)
            # Start with configured tools
            self._tools_list = configured_tools.copy()
        else:
            self._tools_list = []

        # Add all agent tools, avoiding duplicates
        for tool in agent_tools:
            # Check if a tool with this name already exists
            if not any(existing.name == tool.name for existing in self._tools_list):
                self._tools_list.append(tool)

        logger.info(
            f"Structured LLMHost initialized with {len(agent_tools)} agent tools "
            f"from {len(self._participants)} participants, "
            f"total tools: {len(self._tools_list)}"
        )

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
                logger.debug(f"Manager message received with empty content, skipping")
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
                        "participants": list(self._participants.keys())
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
                    f"LLM response processed. Result type: {type(result)}"
                )

                if isinstance(result, AgentTrace) and result.outputs:
                    # Make step request from AgentTrace outputs
                    if isinstance(result.outputs, dict) and "tool_code" in result.outputs:
                        tool_name = result.outputs.get("tool_code", "")
                        parameters = result.outputs.get("parameters", {})

                        # Find the agent role that owns this tool
                        agent_role = None
                        for role, description in self._participants.items():
                            # Check if we have specific tools for this role
                            if hasattr(self, '_participant_tools') and role in self._participant_tools:
                                # Check if any of the role's tools match the requested tool
                                for tool_dict in self._participant_tools[role]:
                                    if tool_dict['name'].lower() == tool_name.lower():
                                        agent_role = role
                                        break
                            # Also check default tool pattern
                            elif f"call_{role.lower()}" == tool_name.lower():
                                agent_role = role
                                break
                            if agent_role:
                                break

                        if agent_role:
                            # Create StepRequest for the tool call
                            step_request = StepRequest(
                                role=agent_role,
                                inputs={
                                    "tool": tool_name,
                                    "tool_inputs": parameters
                                }
                            )
                            logger.info(
                                f"Host {self.agent_name} handling tool_code call: "
                                f"{agent_role}.{tool_name} with parameters: {parameters}"
                            )
                            await self._proposed_step.put(step_request)
                        else:
                            logger.warning(f"No agent found for tool: {tool_name}. Available participants: {list(self._participants.keys())}")
                            # Pass response to manager if no agent found
                            await self.callback_to_groupchat(result)
                    else:
                        # Otherwise just pass the response to the manager
                        await self.callback_to_groupchat(result)

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken,
        public_callback: Callable,
        message_callback: Callable,
        **kwargs: Any,
    ) -> OOBMessages | None:
        """Handle special events and messages, including ConductorRequest.
        
        Override the parent class to rebuild tools when participants change.
        """
        # Call parent class handler first
        result = await super()._handle_events(
            message, cancellation_token, public_callback, message_callback, **kwargs
        )

        # Handle ConductorRequest specifically to rebuild tools
        if isinstance(message, ConductorRequest):
            # Check if participants changed
            old_participants = set(self._participants.keys())
            new_participants = set(message.participants.keys())

            if old_participants != new_participants:
                logger.info(
                    f"Participants changed from {old_participants} to {new_participants}, "
                    f"rebuilding agent tools"
                )
                # Update participants and participant tools
                self._participants.update(message.participants)
                if hasattr(message, 'participant_tools'):
                    self._participant_tools = message.participant_tools

                # Rebuild the tools with new participants
                await self._build_agent_tools()

        return result
