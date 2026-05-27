"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
from collections.abc import AsyncGenerator

import pydantic
from buttermilk._core.runtime_types import MessageContext, message_handler
from buttermilk._core.tool_types import CancellationToken, Tool
from buttermilk._core.messages import FunctionCall, LLMMessage

from buttermilk import AgentInput, StepRequest, bm, logger
from buttermilk._core.agent import UserResponseMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentOutput, ErrorEvent
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import CreateResult, ModelOutput
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent


class StructuredLLMHostAgent(HostAgent, LLMAgent):
    """Host agent that uses structured tool definitions for agent coordination.

    This agent uses an LLM to select which agents to invoke based on structured
    tool definitions provided by agents. The main difference from the base HostAgent
    is that this uses an LLM to dynamically select agents instead of following
    a predefined sequence.
    """

    def __init__(self, **kwargs):
        """Initialize StructuredLLMHostAgent."""
        super().__init__(**kwargs)

        # Initialize attributes specific to this class
        self._user_feedback: list[str] = []

        # Override the output model - we don't need CallOnAgent anymore
        self.output_model = None  # Let the LLM use tool calling directly

    def _clear_pending_steps(self) -> None:
        """Clear all pending steps from the queue."""
        while not self._proposed_step.empty():
            try:
                self._proposed_step.get_nowait()
            except asyncio.QueueEmpty:
                break

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

    @message_handler
    async def _receive_instructions(
        self,
        message: UserResponseMessage,
        ctx: MessageContext,
    ) -> None:
        """Listen to messages and use structured tools to determine next steps."""
        # Messages are now automatically added to context by the base class handle_manager_message method
        # No need to manually call it here since we're overriding the handler

        # Wait for tool schemas to be populated if they haven't been yet
        # This handles the race condition where UserResponseMessage arrives before ConductorRequest processing completes
        max_wait = 5  # seconds
        wait_interval = 0.1
        waited = 0

        while waited < max_wait:
            if self._tools:
                break
            await asyncio.sleep(wait_interval)
            waited += wait_interval

        # Log tool schema status
        if not self._tools:
            msg = f"StructuredLLMHost {self.agent_name} has no tools available after waiting {max_wait}s. This may indicate the participants have not advertised their capabilities."
            logger.error(msg)
            # Send as ErrorEvent for error broadcasting
            error_event = ErrorEvent(
                source=self.agent_id,
                content="Unable to process request: no tools available.",
            )
            await self._publish(error_event)
            return  # Skip processing if no tools are available

        # Skip command messages
        if message.content and str(message.content).startswith(COMMAND_SYMBOL):
            return

        # Skip empty messages
        if not message.content:
            logger.debug("Manager message received with empty content, skipping")
            return

        # Clear any pending steps since the manager has a new request
        self._clear_pending_steps()

        logger.debug(f"Manager interrupted with new request: {message.content}")

        # Use the LLM with structured tools to determine next step
        # The template should be configured to use tool calling
        result = await self._process(
            message=AgentInput(
                inputs={
                    "user_feedback": self._user_feedback,
                    "prompt": str(message.content),
                },
            ),
            cancellation_token=ctx.cancellation_token,  # Pass as kwarg
        )

        if result:
            # Send the response back to the group chat
            await self._publish(result)

    async def _call_llm(
        self,
        messages: list[LLMMessage],
        tools: list[Tool],
        schema: type[pydantic.BaseModel] | None,
        cancellation_token: CancellationToken | None,
    ) -> CreateResult | ModelOutput:
        """Override to intercept tool calls without executing them.

        This allows the StructuredLLMHostAgent to handle tool routing specially.
        """
        # Get the appropriate LLM client
        model_client = bm.llms.get_client(self.parameters["model"])

        # Deduplicate tools by name (handle both Tool objects and ToolSchema dicts)
        def get_tool_name(tool):
            if hasattr(tool, "name"):
                return tool.name  # Tool object
            return tool["name"]  # ToolSchema dict

        tools_list = list({get_tool_name(tool): tool for tool in tools}.values())

        logger.debug(f"StructuredLLMHost calling LLM with {len(tools_list)} tools: {[get_tool_name(tool) for tool in tools_list]}")

        # Use intercept_tools=True to get FunctionCall objects without execution
        return await model_client.call_chat(
            messages=messages,
            tools_list=tools_list,
            cancellation_token=cancellation_token,
            schema=schema,
            intercept_tools=True,  # This is the key flag
        )

    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput | None:
        """Process the message using the LLM with intercepted tool calls."""
        # Fill template and call LLM
        try:
            inputs = message.inputs or {}
            inputs.update(kwargs)

            llm_messages_to_send = await self._fill_template(
                task_params=message.parameters or {},
                inputs=inputs,
            )
        except Exception as e:
            logger.error(f"StructuredLLMHost '{self.agent_id}': Error during template processing: {e!s}")
            raise ProcessingError(f"Error during template processing: {e!s}") from e

        # Call LLM with intercept flag
        # Extract cancellation_token from kwargs if provided
        cancellation_token = kwargs.get("cancellation_token")
        chat_result = await self._call_llm(
            messages=llm_messages_to_send,
            tools=self._tools,
            schema=self.output_model,
            cancellation_token=cancellation_token,
        )

        # Check if we got tool calls in the output
        if isinstance(chat_result.content, list) and all(isinstance(c, FunctionCall) for c in chat_result.content):
            tool_calls: list[FunctionCall] = chat_result.content
            logger.debug(f"StructuredLLMHost received {len(tool_calls)} tool calls from LLM")

            # Use the base class helper to route tool calls
            await self._route_tool_calls_to_agents(tool_calls)

            # Create a more informative summary of the tool calls
            summary = self._create_tool_call_summary(tool_calls)

            # Return a descriptive acknowledgment
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=summary,
                metadata={"tool_calls": len(tool_calls)},
            )

        # If no tool calls, return the LLM response as usual
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=chat_result.content,
            metadata={
                "model": self.parameters["model"],
                "finish_reason": chat_result.finish_reason,
                "usage": getattr(chat_result, "usage", None),
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
                if "message" in args or "content" in args:
                    msg = args.get("message", args.get("content", ""))
                    return f"Processing: {str(msg)[:50]}{'...' if len(str(msg)) > 50 else ''}"
                if "target" in args:
                    return f"Targeting {args['target']} with {tool_name}"
                # Generic single tool call
                return f"Calling {tool_name}"
            except:
                return f"Calling {tool_name}"

        # Multiple tool calls - group by type if possible
        tool_names = [call.name for call in tool_calls]
        unique_tools = list(dict.fromkeys(tool_names))  # Preserve order while removing duplicates

        if len(unique_tools) == 1:
            return f"Making {len(tool_calls)} {unique_tools[0]} calls"
        if len(unique_tools) <= 3:
            return f"Calling: {', '.join(unique_tools)}"
        return f"Orchestrating {len(tool_calls)} tool calls across {len(unique_tools)} tools"

    async def wait_check_current_step_completions(self) -> bool:
        """Override to disable error threshold logic for structured LLM hosts.

        Unlike sequence-based hosts, structured LLM hosts make dynamic decisions
        about which agents to call and should not terminate flows based on error rates.
        Individual agent failures are part of the LLM's decision-making process.

        Returns:
            bool: Always True, unless manually halted by user.
        """
        # Wait for pending tasks to complete but don't check error thresholds
        await self._wait_for_all_tasks_complete()

        # Clear error tracking for the next step (but don't evaluate thresholds)
        async with self._tasks_condition:
            total_failed = sum(self._failed_tasks_by_agent.values())
            if total_failed > 0:
                logger.info(
                    f"StructuredLLMHost {self.agent_id}: {total_failed}/{self._total_tasks_in_step} tasks failed "
                    f"but continuing (no error threshold for LLM-driven flows)"
                )
            self._failed_tasks_by_agent.clear()
            self._total_tasks_in_step = 0

        logger.debug("Current step completed, clear to proceed.")
        return True
