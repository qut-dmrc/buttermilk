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
from buttermilk._core.contract import AgentTrace, GroupchatMessageTypes
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
    
    async def _initialize(self, callback_to_groupchat: Any) -> None:
        """Initialize the host with structured tool definitions from participants."""
        await super()._initialize(callback_to_groupchat=callback_to_groupchat)
        
        # Collect tool definitions from all participant agents
        agent_tools = []
        
        for role, agent in self._participants.items():
            # Skip non-agent participants
            if not hasattr(agent, 'get_tool_definitions'):
                logger.debug(f"Participant {role} does not support tool definitions")
                continue
            
            # Get tool definitions from the agent
            tool_defs = agent.get_tool_definitions()
            
            if not tool_defs:
                # If agent has no explicit tools, create a default one
                logger.debug(f"Creating default tool for agent {role}")
                default_tool = AgentToolDefinition(
                    name=f"call_{role.lower()}",
                    description=f"Send a request to the {role} agent",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The request or question for the agent"
                            }
                        },
                        "required": ["prompt"]
                    },
                    output_schema={"type": "object"}
                )
                tool_defs = [default_tool]
            
            # Create FunctionTool for each tool definition
            for tool_def in tool_defs:
                # Create a closure to capture the role
                async def create_agent_tool(role=role, tool_name=tool_def.name):
                    async def call_agent_tool(**kwargs) -> None:
                        """Call a specific tool on an agent."""
                        # Create StepRequest with tool information
                        choice = StepRequest(
                            role=role,
                            inputs={
                                "tool": tool_name,
                                "tool_inputs": kwargs
                            }
                        )
                        logger.info(
                            f"Host {self.agent_name} calling {role}.{tool_name} "
                            f"with inputs: {kwargs}"
                        )
                        await self.callback_to_groupchat(choice)
                    return call_agent_tool
                
                # Create the actual tool function
                tool_func = await create_agent_tool(role, tool_def.name)
                
                # Create FunctionTool with proper schema
                function_tool = FunctionTool(
                    func=tool_func,
                    name=f"{role.lower()}.{tool_def.name}",
                    description=tool_def.description,
                    # Note: FunctionTool will introspect the function signature,
                    # but we could enhance this to use tool_def.input_schema
                )
                
                agent_tools.append(function_tool)
                logger.debug(f"Registered tool: {role.lower()}.{tool_def.name}")
        
        # Initialize tools list
        self._tools_list = []
        if self.tools:
            # Add any configured tools
            from buttermilk.utils._tools import create_tool_functions
            self._tools_list = create_tool_functions(self.tools)
        
        # Add all agent tools
        self._tools_list.extend(agent_tools)
        
        logger.info(
            f"Structured LLMHost initialized with {len(agent_tools)} agent tools "
            f"from {len(self._participants)} participants"
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
                # The LLM should have called a tool, which will have already
                # sent the appropriate StepRequest via the tool's callback
                logger.debug(
                    f"LLM response processed. Result type: {type(result)}"
                )
                
                # If for some reason we get a direct response without tool use,
                # handle it gracefully
                if isinstance(result, AgentTrace) and result.outputs:
                    # Check if it's requesting to end
                    output_content = str(result.outputs)
                    if "END" in output_content.upper() or "DONE" in output_content.upper():
                        await self._proposed_step.put(StepRequest(role=END))
                    else:
                        # Otherwise just pass the response to the manager
                        await self.callback_to_groupchat(result)