"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import CreateResult
from autogen_core.tools import ToolSchema
from pydantic import PrivateAttr

from buttermilk import buttermilk as bm
from buttermilk._core import AgentInput, StepRequest, logger
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentAnnouncement, AgentOutput, AgentTrace, ErrorEvent, GroupchatMessageTypes
from buttermilk._core.exceptions import ProcessingError
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
    _tool_schemas: list[ToolSchema] = PrivateAttr(default_factory=list)

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
        """Build tool schemas from agent-provided tool definitions.
        
        This method collects tool schemas from agents but does NOT create executable tools.
        Instead, it stores the schemas to pass to the LLM, and we'll handle tool calls
        by converting them to StepRequests.
        """
        tool_schemas = []

        # Collect tool schemas from agent announcements
        for agent_id, announcement in self._agent_registry.items():
            agent_role = announcement.agent_config.role

            if tool_def := announcement.tool_definition:
                # Store the schema directly - no wrapping needed
                tool_schemas.append(tool_def)
                logger.debug(f"Registered tool schema: {tool_def['name']} for {agent_role}")

        # Store schemas separately from executable tools
        self._tool_schemas = tool_schemas

        # Initialize executable tools from config (if any)
        if self.tools:
            configured_tools = create_tool_functions(self.tools)
            self._tools_list = configured_tools
        else:
            self._tools_list = []

        logger.info(
            f"Structured LLMHost collected {len(tool_schemas)} tool schemas "
            f"from {len(self._agent_registry)} announced agents"
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
    
    async def _process(self, *, message: AgentInput,
        cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput:
        """Override to handle tool calls by routing them to agents instead of executing.
        
        This override bypasses the automatic tool execution in the parent class.
        Instead, when the LLM calls a tool, we convert it to a StepRequest and
        queue it for the appropriate agent to handle.
        """
        logger.debug(f"StructuredLLMHost '{self.agent_name}' starting _process")

        # Fill the template as usual
        try:
            llm_messages_to_send = await self._fill_template(
                task_params=message.parameters or {},
                inputs=message.inputs or {},
                context=message.context,
                records=message.records,
            )
        except Exception as e:
            logger.error(f"StructuredLLMHost '{self.agent_id}': Error during template processing: {e!s}")
            error_event = ErrorEvent(source=self.agent_id, content=str(e))
            return AgentOutput(agent_id=self.agent_id, metadata={"error": True}, outputs=error_event)

        # Get the LLM client
        model_client = bm.llms.get_autogen_chat_client(self._model)
        
        # Call create() directly with tool schemas (not executable tools)
        # This returns FunctionCall objects without executing them
        logger.debug(f"StructuredLLMHost calling LLM with {len(self._tool_schemas)} tool schemas")
        
        try:
            create_result: CreateResult = await model_client.create(
                messages=llm_messages_to_send,
                tools=self._tool_schemas,  # Pass schemas, not executable tools
                cancellation_token=cancellation_token,
            )
        except Exception as llm_error:
            msg = f"StructuredLLMHost {self.agent_id}: Error during LLM call: {llm_error}"
            logger.error(msg)
            raise ProcessingError(msg) from llm_error

        # Check if the LLM returned tool calls
        if isinstance(create_result.content, list) and all(isinstance(c, FunctionCall) for c in create_result.content):
            tool_calls: list[FunctionCall] = create_result.content
            logger.info(f"StructuredLLMHost received {len(tool_calls)} tool calls from LLM")
            
            # Convert each tool call to a StepRequest
            for call in tool_calls:
                # Find which agent handles this tool
                agent_role = None
                for agent_id, announcement in self._agent_registry.items():
                    if announcement.tool_definition and announcement.tool_definition['name'] == call.name:
                        agent_role = announcement.agent_config.role
                        break
                
                if not agent_role:
                    logger.warning(f"No agent found for tool: {call.name}")
                    continue
                
                # Parse the arguments
                import json
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool arguments: {call.arguments}")
                    continue
                
                # Create StepRequest for this tool call
                step_request = StepRequest(
                    role=agent_role,
                    inputs=arguments,
                    metadata={"tool_name": call.name, "tool_call_id": call.id}
                )
                
                logger.info(f"StructuredLLMHost routing tool call {call.name} to {agent_role}")
                await self._proposed_step.put(step_request)
            
            # Return a simple acknowledgment
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=f"Routing {len(tool_calls)} tool calls to agents",
                metadata={"tool_calls": len(tool_calls)}
            )
        
        # If no tool calls, return the LLM response as usual
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=create_result.content,
            metadata={
                "model": self._model,
                "finish_reason": create_result.finish_reason,
                "usage": create_result.usage.model_dump() if create_result.usage else None
            }
        )
