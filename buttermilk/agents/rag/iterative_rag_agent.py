"""Iterative RAG agent that uses composition over inheritance.

This module provides a RAG agent that:
- Uses external search tools instead of embedded logic
- Guarantees structured outputs with citations
- Relies on templates for orchestration
- Is specifically designed for iterative search
"""

from typing import Any

from pydantic import Field
import json

from buttermilk.agents.rag.simple_rag_agent import RagAgent, ResearchResult
from buttermilk._core.contract import AgentInput, AgentOutput, ErrorEvent
from buttermilk._core.exceptions import ProcessingError
from buttermilk import logger
from autogen_core.models import AssistantMessage


class IterativeRagAgent(RagAgent):
    """RAG agent that performs iterative searches.

    This agent is designed to:
    - Perform an initial search.
    - Evaluate the results.
    - Perform additional searches to refine the results.
    - Synthesize a final answer from the accumulated results.
    """

    template: str = Field("iterative_rag", description="Jinja2 template to use for this agent.")

    async def _process(self, *, message: AgentInput, cancellation_token=None, **kwargs) -> AgentOutput:
        """Core processing logic for iterative RAG.

        This method implements a loop that repeatedly calls the LLM, executes tools,
        and processes results until a final answer is synthesized or max iterations are reached.
        """
        logger.debug(f"IterativeRagAgent '{self.agent_name}' starting _process for message_id: {getattr(message, 'message_id', 'N/A')}.")

        max_iterations = self.parameters.get("max_iterations", 5)  # Configurable max iterations
        current_iteration = 0
        chat_history = message.context  # Start with initial context
        current_input_message = message.inputs.get("prompt", "")

        while current_iteration < max_iterations:
            current_iteration += 1
            logger.info(f"IterativeRagAgent: Iteration {current_iteration}/{max_iterations}")

            # Prepare messages for the LLM
            llm_messages_to_send = await self._fill_template(
                task_params=message.parameters,
                inputs={"prompt": current_input_message},  # Pass current prompt to template
                context=chat_history,
                records=message.records,
            )

            # Get the appropriate AutoGenWrapper instance
            import buttermilk

            model_client = buttermilk.get_bm().llms.get_autogen_chat_client(self.parameters["model"])

            try:
                chat_result = await model_client.call_chat(
                    messages=llm_messages_to_send,
                    tools_list=self._tools_list,
                    cancellation_token=cancellation_token,
                    schema=self._output_model,  # Pass expected Pydantic schema for structured output
                )
            except Exception as llm_error:
                msg = f"Agent {self.agent_id}: Error during LLM call: {llm_error}"
                logger.error(msg, exc_info=True)
                return AgentOutput(agent_id=self.agent_id, outputs=ErrorEvent(source=self.agent_id, content=msg))

            # Add LLM's response to chat history
            chat_history.append(
                chat_result.to_llm_message()
                if hasattr(chat_result, "to_llm_message")
                else AssistantMessage(content=str(chat_result), source="IterativeRagAgent")
            )

            # Check for tool calls
            if hasattr(chat_result, "tool_calls") and chat_result.tool_calls:
                logger.info(f"IterativeRagAgent: LLM requested tool calls: {len(chat_result.tool_calls)}")
                tool_outputs = []
                for tool_call in chat_result.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    if tool_name not in self.tools:
                        error_msg = f"Tool '{tool_name}' not found for agent '{self.agent_name}'."
                        logger.error(error_msg)
                        tool_outputs.append(ErrorEvent(source=self.agent_id, content=error_msg))
                        continue

                    try:
                        # Execute the tool
                        tool_instance = self.tools[tool_name]
                        # Assuming tool_args is a JSON string, parse it
                        parsed_tool_args = json.loads(tool_args)
                        tool_result = await tool_instance.run(**parsed_tool_args)  # Assuming tools have a .run method
                        tool_outputs.append(tool_result)
                        logger.info(f"Tool '{tool_name}' executed. Result: {tool_result.content[:100]}...")
                    except Exception as tool_error:
                        error_msg = f"Error executing tool '{tool_name}': {tool_error}"
                        logger.error(error_msg, exc_info=True)
                        tool_outputs.append(ErrorEvent(source=self.agent_id, content=error_msg))

                # Add tool outputs to chat history for next LLM call
                for output in tool_outputs:
                    chat_history.append(
                        output.to_llm_message()
                        if hasattr(output, "to_llm_message")
                        else AssistantMessage(content=str(output), source="IterativeRagAgent")
                    )

                # Update current_input_message with tool results for the next iteration's prompt
                current_input_message = "Tool results: " + "; ".join([str(o.content) for o in tool_outputs if hasattr(o, "content")])

            elif chat_result.finish_reason == "stop":
                logger.info(f"IterativeRagAgent: LLM synthesized final answer. Iterations: {current_iteration}")
                # Parse the final result using the agent's output model
                if self._output_model:
                    try:
                        parsed_output = self._output_model.model_validate_json(chat_result.content)
                        return AgentOutput(agent_id=self.agent_id, outputs=parsed_output, metadata=chat_result.model_dump())
                    except Exception as parse_error:
                        msg = f"Failed to parse final LLM response into {self._output_model.__name__}: {parse_error}"
                        logger.error(msg, exc_info=True)
                        return AgentOutput(agent_id=self.agent_id, outputs=ErrorEvent(source=self.agent_id, content=msg))
                else:
                    return AgentOutput(agent_id=self.agent_id, outputs=chat_result.content, metadata=chat_result.model_dump())
            else:
                logger.warning(f"IterativeRagAgent: LLM finished with unexpected reason: {chat_result.finish_reason}. Content: {chat_result.content}")
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs=ErrorEvent(source=self.agent_id, content=f"LLM finished with unexpected reason: {chat_result.finish_reason}"),
                )

        logger.warning(f"IterativeRagAgent: Max iterations ({max_iterations}) reached without a final answer.")
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=ErrorEvent(source=self.agent_id, content=f"Max iterations ({max_iterations}) reached without a final answer."),
        )
