"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import CreateResult
from pydantic import PrivateAttr

from buttermilk import buttermilk as bm
from buttermilk._core import AgentInput, StepRequest, logger
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentOutput, ErrorEvent, GroupchatMessageTypes
from buttermilk._core.exceptions import ProcessingError
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent
from buttermilk.utils._tools import create_tool_functions


class StructuredLLMHostAgent(HostAgent, LLMAgent):
    """Host agent that uses structured tool definitions for agent coordination.

    This agent uses an LLM to select which agents to invoke based on structured
    tool definitions provided by agents. The main difference from the base HostAgent
    is that this uses an LLM to dynamically select agents instead of following
    a predefined sequence.
    """

    _user_feedback: list[str] = PrivateAttr(default_factory=list)

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

        # Initialize empty tools list
        if self.tools:
            self._tools_list = create_tool_functions(self.tools)
        else:
            self._tools_list = []

        logger.debug(f"StructuredLLMHost {self.agent_name} initialized with {len(self._tools_list)} configured tools.")

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute.
        
        Unlike the base class which follows a predefined sequence,
        this implementation uses a queue-based approach where the LLM
        decides which agent to invoke next.
        """
        # First, say hello to the user
        await asyncio.sleep(3)  # Let the group chat initialize
        yield StepRequest(
            role=MANAGER,
            content="Hi! What would you like to do?",
        )

        while True:
            # Wait for the _listen method (via LLM) to add a proposed step to the queue
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
                    }
                ),
                public_callback=public_callback,
                cancellation_token=cancellation_token,
                **kwargs,
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
        model_client = bm.llms.get_autogen_chat_client(self.parameters["model"])

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

            # Use the base class helper to route tool calls
            await self._route_tool_calls_to_agents(tool_calls)

            # Create a more informative summary of the tool calls
            summary = self._create_tool_call_summary(tool_calls)

            # Return a descriptive acknowledgment
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=summary,
                metadata={"tool_calls": len(tool_calls)}
            )

        # If no tool calls, return the LLM response as usual
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=create_result.content,
            metadata={
                "model": self.parameters["model"],
                "finish_reason": create_result.finish_reason,
                "usage": create_result.usage if create_result.usage else None,
            },
        )

    def _create_tool_call_summary(self, tool_calls: list[FunctionCall]) -> str:
        """Create a human-readable summary of tool calls.
        
        Args:
            tool_calls: List of FunctionCall objects
            
        Returns:
            str: A concise summary of what tools are being called
        """
        if not tool_calls:
            return "No tool calls requested"

        if len(tool_calls) == 1:
            call = tool_calls[0]
            # Try to extract a meaningful description from the tool name and arguments
            tool_name = call.name

            # Try to parse arguments for key information
            try:
                import json
                args = json.loads(call.arguments)

                # Common patterns for better descriptions
                if "query" in args:
                    return f"Searching for: {args['query'][:50]}{'...' if len(str(args['query'])) > 50 else ''}"
                elif "message" in args or "content" in args:
                    msg = args.get("message", args.get("content", ""))
                    return f"Processing: {str(msg)[:50]}{'...' if len(str(msg)) > 50 else ''}"
                elif "target" in args:
                    return f"Targeting {args['target']} with {tool_name}"
                else:
                    # Generic single tool call
                    return f"Calling {tool_name}"
            except:
                return f"Calling {tool_name}"

        # Multiple tool calls - group by type if possible
        tool_names = [call.name for call in tool_calls]
        unique_tools = list(dict.fromkeys(tool_names))  # Preserve order while removing duplicates

        if len(unique_tools) == 1:
            return f"Making {len(tool_calls)} {unique_tools[0]} calls"
        elif len(unique_tools) <= 3:
            return f"Calling: {', '.join(unique_tools)}"
        else:
            return f"Orchestrating {len(tool_calls)} tool calls across {len(unique_tools)} tools"
