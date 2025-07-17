"""Iterative RAG agent that uses composition over inheritance.

This module provides a RAG agent that:
- Uses external search tools instead of embedded logic
- Guarantees structured outputs with citations
- Relies on templates for orchestration
- Is specifically designed for iterative search
"""

import json

from autogen_core.models import AssistantMessage

from buttermilk import logger
from buttermilk._core.contract import AgentInput, AgentOutput, ErrorEvent, ToolOutput
from buttermilk.agents.rag.simple_rag_agent import RagAgent


class IterativeRagAgent(RagAgent):
    """RAG agent that performs iterative searches.

    This agent is designed to:
    - Perform an initial search.
    - Evaluate the results.
    - Perform additional searches to refine the results.
    - Synthesize a final answer from the accumulated results.
    """

    def __init__(self, **kwargs):
        """Initialize IterativeRagAgent with template configuration."""
        super().__init__(**kwargs)

        # Template configuration - moved from Field declaration
        self.template: str = kwargs.get("template", "iterative_rag")

    async def _process(self, *, message: AgentInput, cancellation_token=None, **kwargs) -> AgentOutput:
        """Core processing logic for iterative RAG.

        This method implements an iterative RAG cycle:
        1. Generate tool calls -> run tools -> reflect and generate new tool calls
        2. Repeat until exhausted or max iterations
        3. Reflect and synthesise final result

        After executing tools, we give the LLM a chance to reflect
        on the results and decide whether to continue with more tool calls or synthesize.
        """
        logger.debug(f"IterativeRagAgent '{self.agent_name}' starting _process for message_id: {getattr(message, 'message_id', 'N/A')}.")

        max_iterations = self.parameters.get("max_iterations", 5)  # Configurable max iterations
        current_iteration = 0
        chat_history = list(message.context) if message.context else []  # Start with initial context
        current_input_message = message.inputs.get("prompt", "")
        initial_prompt = message.inputs.get("prompt", "")

        # Prepare initial messages for the LLM
        llm_messages_to_send = await self._fill_template(
            task_params=message.parameters,
            inputs={"prompt": initial_prompt},
            context=chat_history,
            records=message.records,
        )

        # Get the appropriate AutoGenWrapper instance
        import buttermilk

        model_client = buttermilk.get_bm().llms.get_autogen_chat_client(self.parameters["model"])

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
                    tools_list=self._tools,
                    cancellation_token=cancellation_token,
                    schema=self._output_model,  # Pass expected Pydantic schema for structured output
                )
            except Exception as llm_error:
                msg = f"Agent {self.agent_id}: Error during LLM call: {llm_error}"
                logger.error(msg, exc_info=True)
                return AgentOutput(agent_id=self.agent_id, outputs=ErrorEvent(source=self.agent_id, content=msg))

            # Add LLM's response to chat history
            chat_history.append(
                (
                    chat_result.to_llm_message()
                    if hasattr(chat_result, "to_llm_message")
                    else AssistantMessage(content=str(chat_result), source="IterativeRagAgent")
                ),
            )

            # Check for tool calls
            if hasattr(chat_result, "tool_calls") and chat_result.tool_calls:
                logger.info(f"IterativeRagAgent: LLM requested tool calls: {len(chat_result.tool_calls)}")
                tool_outputs = []

                for tool_call in chat_result.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    # Find the tool by name in our tools list
                    matching_tool = None
                    for tool in self._tools:
                        if hasattr(tool, "name") and tool.name == tool_name:
                            matching_tool = tool
                            break

                    if matching_tool is None:
                        error_msg = f"Tool '{tool_name}' not found for agent '{self.agent_name}'."
                        logger.error(error_msg)
                        tool_outputs.append(ErrorEvent(source=self.agent_id, content=error_msg))
                        continue

                    try:
                        # Parse tool arguments
                        if isinstance(tool_args, str):
                            parsed_tool_args = json.loads(tool_args)
                        else:
                            parsed_tool_args = tool_args

                        # Execute the tool using the func method from FunctionTool
                        if hasattr(matching_tool, "func"):
                            tool_result = await matching_tool.func(**parsed_tool_args)
                        elif hasattr(matching_tool, "run"):
                            tool_result = await matching_tool.run(**parsed_tool_args)
                        else:
                            raise AttributeError(f"Tool {tool_name} has no callable func or run method")

                        # Convert result to ToolOutput if it isn't already
                        if not isinstance(tool_result, ToolOutput):
                            tool_result = ToolOutput(
                                name=tool_name,
                                call_id=getattr(tool_call, "id", ""),
                                content=str(tool_result),
                                results=tool_result if not isinstance(tool_result, str) else None,
                            )

                        tool_outputs.append(tool_result)
                        logger.info(f"Tool '{tool_name}' executed. Result: {tool_result.content[:100]}...")
                    except Exception as tool_error:
                        error_msg = f"Error executing tool '{tool_name}': {tool_error}"
                        logger.error(error_msg, exc_info=True)
                        tool_outputs.append(ErrorEvent(source=self.agent_id, content=error_msg))

                # Add tool outputs to chat history
                for output in tool_outputs:
                    chat_history.append(
                        (
                            output.to_llm_message()
                            if hasattr(output, "to_llm_message")
                            else AssistantMessage(content=str(output), source="IterativeRagAgent")
                        ),
                    )

                # After executing tools, prepare messages for the next LLM call
                # to allow reflection on tool results. Use the updated chat_history which now
                # includes the tool outputs, and let the LLM decide what to do next.
                llm_messages_to_send = chat_history

                # Continue the loop to give LLM a chance to reflect on tool results
                # and decide whether to make more tool calls or synthesize final result
                continue

            if chat_result.finish_reason == "stop":
                logger.info(f"IterativeRagAgent: LLM decided to synthesize final answer. Iterations: {current_iteration}")

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
                # Don't immediately exit - give it another chance unless we're at max iterations
                llm_messages_to_send = chat_history
                continue

        logger.warning(f"IterativeRagAgent: Max iterations ({max_iterations}) reached. Attempting final synthesis.")

        # If max iterations reached, make one final call to synthesize results
        try:
            final_result = await model_client.call_chat(
                messages=chat_history,
                tools_list=[],  # No tools for final synthesis
                cancellation_token=cancellation_token,
                schema=self._output_model,
            )

            if self._output_model:
                try:
                    parsed_output = self._output_model.model_validate_json(final_result.content)
                    return AgentOutput(agent_id=self.agent_id, outputs=parsed_output, metadata=final_result.model_dump())
                except Exception as parse_error:
                    logger.error(f"Failed to parse final synthesis: {parse_error}")
                    return AgentOutput(agent_id=self.agent_id, outputs=final_result.content, metadata=final_result.model_dump())
            else:
                return AgentOutput(agent_id=self.agent_id, outputs=final_result.content, metadata=final_result.model_dump())

        except Exception as final_error:
            logger.error(f"Error during final synthesis: {final_error}")
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=ErrorEvent(
                    source=self.agent_id, content=f"Max iterations ({max_iterations}) reached and final synthesis failed: {final_error}"
                ),
            )
