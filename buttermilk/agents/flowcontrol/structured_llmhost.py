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
    _step_results: dict[str, asyncio.Future] = PrivateAttr(default_factory=dict)
    _pending_tool_calls: dict[str, dict] = PrivateAttr(default_factory=dict)  # Track by role + timestamp
    _testing_mode: bool = PrivateAttr(default=False)  # For tests - don't wait for results

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
    
    def set_testing_mode(self, enabled: bool = True) -> None:
        """Enable or disable testing mode.
        
        In testing mode, tool calls return immediately with 'queued' status
        instead of waiting for actual results. This is useful for unit tests.
        """
        self._testing_mode = enabled
        
    def set_participant_tools_for_testing(self, participant_tools: dict[str, Any]) -> None:
        """Set participant tools for testing purposes.
        
        This method allows tests to populate the _participant_tools directly
        without going through the full orchestrator initialization.
        """
        self._participant_tools = participant_tools
        
    def _cleanup_expired_futures(self) -> None:
        """Clean up any futures that have been hanging around too long."""
        import time
        current_time = time.time()
        expired_keys = []
        
        for step_id, future in self._step_results.items():
            if future.done():
                expired_keys.append(step_id)
            # Also check if future has been waiting too long (more than 60 seconds)
            elif hasattr(future, '_created_at'):
                if current_time - future._created_at > 60:
                    expired_keys.append(step_id)
                    if not future.done():
                        future.cancel()
        
        for key in expired_keys:
            self._step_results.pop(key, None)
            
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired step futures")

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

        This simplified method creates FunctionTool objects from both:
        1. Agent announcements (real-time updates)
        2. Initial participant tools (from ConductorRequest)

        This ensures tools are available even before all agents have announced.
        """
        agent_tools = []

        # First, build tools from agent announcements (most up-to-date)
        for agent_id, announcement in self._agent_registry.items():
            if not announcement.tool_definition:
                logger.debug(f"No tool definition provided by agent {agent_id}, skipping")
                continue

            tool_def = announcement.tool_definition
            agent_role = announcement.agent_config.role

            # Create closure to capture agent role
            def create_tool_func(target_role: str, tool_name: str):
                async def call_agent(**kwargs: Any) -> dict[str, Any]:
                    """Call an agent using its tool definition."""
                    import shortuuid
                    
                    # Generate unique step ID
                    step_id = f"step_{shortuuid.uuid()[:8]}"
                    
                    step_request = StepRequest(
                        role=target_role,
                        inputs=kwargs,
                        content=f"Invoking {tool_name} on {target_role}",
                        metadata={"step_id": step_id, "tool_name": tool_name}
                    )
                    
                    # Create future to wait for result
                    result_future = asyncio.Future()
                    result_future._target_role = target_role  # Tag for fallback resolution
                    result_future._created_at = __import__('time').time()  # Track creation time
                    self._step_results[step_id] = result_future
                    
                    logger.info(
                        f"Host {self.agent_name} invoking {target_role} via tool {tool_name} "
                        f"with inputs: {list(kwargs.keys())} (step_id: {step_id})"
                    )
                    
                    # Queue the step request
                    await self._proposed_step.put(step_request)
                    
                    # In testing mode, return immediately without waiting
                    if self._testing_mode:
                        return {"status": "queued", "agent": target_role, "step_id": step_id}
                    
                    # Wait for the actual result
                    try:
                        result = await asyncio.wait_for(result_future, timeout=30.0)
                        return result
                    except asyncio.TimeoutError:
                        logger.error(f"Tool call {tool_name} timed out after 30 seconds")
                        return {"error": "Tool call timed out", "tool": tool_name}
                    finally:
                        # Clean up the future
                        self._step_results.pop(step_id, None)
                        
                return call_agent

            # Create FunctionTool using agent-provided definition
            tool = FunctionTool(
                func=create_tool_func(agent_role, tool_def["name"]),
                name=tool_def["name"],
                description=tool_def["description"]
            )
            agent_tools.append(tool)
            logger.debug(f"Registered tool from announcement: {tool_def['name']} for {agent_role}")

        # Second, build tools from initial participant tools (from ConductorRequest)
        # This ensures we have tools even before agents announce themselves
        if hasattr(self, "_participant_tools") and self._participant_tools:
            for role, tool_definitions in self._participant_tools.items():

                for tool_def in tool_definitions:
                    tool_name = tool_def.get("name", f"call_{role.lower()}")

                    # Skip if we already have this tool from announcements
                    if any(tool.name == tool_name for tool in agent_tools):
                        continue

                    # Create closure for participant tool
                    def create_participant_tool_func(target_role: str, target_tool_name: str):
                        async def call_participant(**kwargs: Any) -> dict[str, Any]:
                            """Call a participant using initial tool definition."""
                            import shortuuid
                            
                            # Generate unique step ID
                            step_id = f"step_{shortuuid.uuid()[:8]}"
                            
                            step_request = StepRequest(
                                role=target_role,
                                inputs=kwargs,
                                content=f"Invoking {target_tool_name} on {target_role}",
                                metadata={"step_id": step_id, "tool_name": target_tool_name}
                            )
                            
                            # Create future to wait for result
                            result_future = asyncio.Future()
                            result_future._target_role = target_role  # Tag for fallback resolution
                            result_future._created_at = __import__('time').time()  # Track creation time
                            self._step_results[step_id] = result_future
                            
                            logger.info(
                                f"Host {self.agent_name} invoking {target_role} via participant tool {target_tool_name} "
                                f"with inputs: {list(kwargs.keys())} (step_id: {step_id})"
                            )
                            
                            # Queue the step request
                            await self._proposed_step.put(step_request)
                            
                            # In testing mode, return immediately without waiting
                            if self._testing_mode:
                                return {"status": "queued", "agent": target_role, "step_id": step_id}
                            
                            # Wait for the actual result
                            try:
                                result = await asyncio.wait_for(result_future, timeout=30.0)
                                return result
                            except asyncio.TimeoutError:
                                logger.error(f"Tool call {target_tool_name} timed out after 30 seconds")
                                return {"error": "Tool call timed out", "tool": target_tool_name}
                            finally:
                                # Clean up the future
                                self._step_results.pop(step_id, None)
                                
                        return call_participant

                    # Create FunctionTool from participant tool definition
                    tool = FunctionTool(
                        func=create_participant_tool_func(role, tool_name),
                        name=tool_name,
                        description=tool_def.get("description", f"Call {role} agent")
                    )
                    agent_tools.append(tool)
                    logger.debug(f"Registered tool from participant_tools: {tool_name} for {role}")

        # Initialize tools list using parent class pattern
        # First, load any configured tools from the agent config
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

    async def _handle_step_completion(self, message: GroupchatMessageTypes) -> None:
        """Handle step completion and resolve waiting futures."""
        # Clean up expired futures first
        self._cleanup_expired_futures()
        
        # Check for AgentTrace or AgentOutput that might contain step results
        step_id = None
        result_data = None
        agent_role = None
        
        if isinstance(message, AgentTrace):
            # Check if this trace has step metadata
            if message.inputs and message.inputs.metadata:
                step_id = message.inputs.metadata.get("step_id")
                if step_id and message.outputs:
                    result_data = message.outputs.outputs if hasattr(message.outputs, 'outputs') else message.outputs
            
            # Also get the agent role for fallback matching
            if message.agent_info and hasattr(message.agent_info, 'role'):
                agent_role = message.agent_info.role
                
        elif isinstance(message, AgentOutput):
            # Check metadata for step ID
            if hasattr(message, 'metadata') and message.metadata:
                step_id = message.metadata.get("step_id")
            
            # Get agent role from the output
            if hasattr(message, 'role'):
                agent_role = message.role
            elif hasattr(message, 'source'):
                # Try to extract role from source
                agent_role = message.source
            
            if message.outputs:
                result_data = message.outputs
        
        # Primary approach: resolve by step_id
        if step_id and step_id in self._step_results:
            future = self._step_results[step_id]
            if not future.done():
                if result_data is not None:
                    future.set_result(result_data)
                    logger.debug(f"Resolved step result for {step_id}")
                else:
                    future.set_result({"status": "completed", "step_id": step_id})
                    logger.debug(f"Resolved step {step_id} with default completion status")
                return
        
        # Fallback approach: resolve by agent role (most recent pending call for this role)
        if agent_role and result_data is not None:
            # Find the most recent pending future for this role
            for step_id, future in list(self._step_results.items()):
                if not future.done():
                    # Check if this future is for the same role
                    # This is a heuristic - we track pending calls by role
                    if hasattr(future, '_target_role') and future._target_role == agent_role:
                        future.set_result(result_data)
                        logger.debug(f"Resolved step result for role {agent_role} (step_id: {step_id})")
                        return

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
        
        # Check for step completion results first
        await self._handle_step_completion(message)
        
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

