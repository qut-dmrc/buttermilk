"""LLM Host agent that uses structured tool definitions instead of natural language.

This is the refactored version of LLMHostAgent that implements Phase 3 of Issue #83.
"""

import asyncio
from collections.abc import AsyncGenerator

from autogen_core import CancellationToken, FunctionCall, MessageContext, message_handler
from autogen_core.models import CreateResult

from buttermilk import buttermilk as bm
from buttermilk._core import AgentInput, StepRequest, logger
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentOutput, ErrorEvent
from buttermilk._core.exceptions import ProcessingError
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
        self._output_model = None  # Let the LLM use tool calling directly

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
        message: ManagerMessage,
        ctx: MessageContext,
    ) -> None:
        """Listen to messages and use structured tools to determine next steps."""
        # TODO: figure out if we need to add user messages to context and fix this by adding a proper add_to_context method in Agent
        # # Save messages to context
        # await super().handle_groupchat_message(
        #     message=message,ctx=ctx
        # )

        # Wait for tool schemas to be populated if they haven't been yet
        # This handles the race condition where ManagerMessage arrives before ConductorRequest processing completes
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
            await self._publish("Unable to process request: no tools available.")
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

        logger.info(f"Manager interrupted with new request: {message.content}")

        # Use the LLM with structured tools to determine next step
        # The template should be configured to use tool calling
        result = await self._process(
            message=AgentInput(
                inputs={
                    "user_feedback": self._user_feedback,
                    "prompt": str(message.content),
                },
            ),
            cancellation_token=ctx.cancellation_token,
        )

        if result:
            # Send the response back to the group chat
            await self._publish(result)

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput:
        """Override to handle tool calls by routing them to agents instead of executing.

        This override bypasses the automatic tool execution in the parent class.
        Instead, when the LLM calls a tool, we convert it to a StepRequest and
        queue it for the appropriate agent to handle.
        """
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

        # deduplicate tools by name
        tools_list = list({tool.name: tool for tool in self._tools}.values())

        # This returns FunctionCall objects without executing them
        logger.debug(f"StructuredLLMHost calling LLM with {len(tools_list)} tools: {[tool.name for tool in tools_list]}")

        try:
            create_result: CreateResult = await model_client.create(
                messages=llm_messages_to_send,
                tools=tools_list,  # Pass Tool objects
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
            return AgentOutput(agent_id=self.agent_id, outputs=summary, metadata={"tool_calls": len(tool_calls)})

        # If no tool calls, return the LLM response as usual
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=create_result.content,
            metadata={
                "model": self.parameters["model"],
                "finish_reason": create_result.finish_reason,
                "usage": create_result.usage or None,
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
