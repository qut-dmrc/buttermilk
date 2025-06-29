"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any, Type

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool, BaseTool
from pydantic import BaseModel, PrivateAttr, create_model, Field

from buttermilk._core import AgentInput, StepRequest, logger
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentAnnouncement, AgentOutput, AgentTrace, GroupchatMessageTypes
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent
from buttermilk.utils._tools import create_tool_functions


class AgentToolWrapper(BaseTool[BaseModel, BaseModel]):
    """Custom tool wrapper for agent invocations that preserves parameter schemas."""
    
    def __init__(
        self,
        agent_role: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        proposed_step_queue: asyncio.Queue[StepRequest],
        host_agent_name: str
    ):
        # Create a Pydantic model from the input schema
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        
        # Build field definitions for the Pydantic model
        field_definitions = {}
        for prop_name, prop_schema in properties.items():
            prop_type = self._json_schema_to_python_type(prop_schema)
            is_required = prop_name in required
            default = ... if is_required else prop_schema.get("default", None)
            description = prop_schema.get("description", "")
            
            field_definitions[prop_name] = (prop_type, Field(default=default, description=description))
        
        # Create the args model dynamically
        args_model = create_model(
            f"{tool_name}_args",
            **field_definitions
        )
        
        # Initialize the base tool
        super().__init__(
            args_type=args_model,
            return_type=BaseModel,  # We return None but need a BaseModel type
            name=tool_name,
            description=tool_description
        )
        
        self.agent_role = agent_role
        self.proposed_step_queue = proposed_step_queue
        self.host_agent_name = host_agent_name
    
    def _json_schema_to_python_type(self, schema: dict[str, Any]) -> Type:
        """Convert JSON schema type to Python type."""
        type_str = schema.get("type", "string")
        
        if type_str == "string":
            if "enum" in schema:
                # For simplicity, return str even for enums
                return str
            return str
        elif type_str == "integer":
            return int
        elif type_str == "number":
            return float
        elif type_str == "boolean":
            return bool
        elif type_str == "array":
            # For simplicity, return list without item type
            return list
        elif type_str == "object":
            # For simplicity, return dict
            return dict
        else:
            return Any
    
    async def run(self, args: BaseModel, cancellation_token: CancellationToken) -> Any:
        """Execute the tool by sending a StepRequest to the target agent."""
        # Convert args model to dict
        inputs = args.model_dump()
        
        step_request = StepRequest(
            role=self.agent_role,
            inputs=inputs,
            metadata={"tool_name": self.name}
        )
        logger.info(f"Host {self.host_agent_name} invoking {self.agent_role} tool {self.name}")
        await self.proposed_step_queue.put(step_request)
        
        # Return empty dict as we don't have a meaningful return value
        return {}


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
        
        This method extracts actual @tool decorated methods from agents and wraps them
        to send StepRequest messages, following the agent-centric approach.
        """
        agent_tools = []

        # Build tools from agent announcements
        for agent_id, announcement in self._agent_registry.items():
            agent_role = announcement.agent_config.role

            if tool_def :=  announcement.tool_definition:
                # Get input schema from tool definition
                input_schema = tool_def.get("input_schema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })

                # Create custom tool wrapper that preserves parameter schemas
                tool = AgentToolWrapper(
                    agent_role=agent_role,
                    tool_name=tool_def["name"],
                    tool_description=tool_def.get("description", f"Call {agent_role} agent"),
                    input_schema=input_schema,
                    proposed_step_queue=self._proposed_step,
                    host_agent_name=self.agent_name
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
