"""LLM-based Agent for Buttermilk.

This module provides the `LLMAgent` class, a foundational agent that integrates
with large language models (LLMs) to process tasks. It extends the base `Agent`
class with the ability to:
1. Fill prompt templates with input data and conversation context
2. Call LLMs with the rendered prompts
3. Parse structured outputs from LLM responses

`LLMAgent` serves as a base class for more specialized agents that require
LLM interaction, handling the core template rendering and LLM communication
workflow.
"""

from collections.abc import AsyncGenerator
from typing import Any, Self

import hydra
import pydantic
from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, LLMMessage, UserMessage
from autogen_core.tools import Tool

from buttermilk import buttermilk as bm, logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput, ErrorEvent
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import CreateResult, ModelOutput
from buttermilk._core.types import Record
from buttermilk.utils._tools import create_tool_functions
from buttermilk.utils.templating import load_template, make_messages


class LLMAgent(Agent):
    """Agent that uses an LLM for text processing and generation.

    `LLMAgent` extends the base `Agent` class to add LLM-powered capabilities.
    It manages the complete workflow of:
    1. Loading and rendering prompt templates with provided data
    2. Communicating with LLMs through the global LLM manager
    3. Parsing and validating LLM responses

    The agent can work with both unstructured text responses and structured
    outputs (when `_output_model` is specified as a Pydantic model).

    Configuration (from `AgentConfig`):
        template: Name of the prompt template to use (required in parameters)
        model: Name of the LLM model to use (e.g., 'gpt-4', 'claude-3')
        temperature: LLM temperature parameter for response variability
        fail_on_unfilled_parameters: Whether to fail if template variables are missing
        tools: List of tools (functions) the agent can use

    Attributes:
        _model (str): The name/identifier of the LLM model this agent uses.
        _output_model (type[pydantic.BaseModel] | None): Optional Pydantic model
            for structured output parsing.
        _tools_list (list[Tool]): List of Autogen-compatible tool objects.
        fail_on_unfilled_parameters (bool): If True, raises an error when
            template variables are missing from inputs.

    Example:
        ```python
        agent = LLMAgent(
            agent_name="Analyzer",
            role="TEXT_ANALYZER",
            parameters={"template": "analysis_prompt", "model": "gpt-4"},
            fail_on_unfilled_parameters=True
        )
        
        result = await agent.invoke(
            AgentInput(inputs={"text": "Hello world"})
        )
        ```

    """

    # LLM-specific configurations
    # These are now initialized in __init__ instead of using PrivateAttr

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an LLMAgent with the provided configuration.

        Extracts the model name from parameters and stores it in `_model`.
        The actual model initialization happens in `init_model()`.

        Args:
            **kwargs: Configuration parameters passed to AgentConfig.
                Must include 'model' in parameters.

        Raises:
            ValueError: If 'model' is not specified in parameters.

        """
        super().__init__(**kwargs)
        if "model" not in self.parameters:
            raise ValueError(f"Agent {self.agent_name}: 'model' is required in agent parameters.")

        # Initialize private attributes
        self._model: str = self.parameters.get("model", "")
        self._output_model: type[pydantic.BaseModel] | None = None
        self._tools: list[Tool] = []

        # Control behavior - moved from Field declaration
        self.fail_on_unfilled_parameters: bool = kwargs.get("fail_on_unfilled_parameters", True)

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        """Loads tool configurations and converts them to Autogen-compatible tools.

        This Pydantic validator runs after the agent model is created.
        It checks `self.tools` (an `AgentConfig` field, typically populated from
        Hydra configuration) and uses `create_tool_functions` to convert these
        tool definitions into a list of Autogen-compatible tool objects
        (`_tools`).

        Returns:
            Self: The agent instance with `_tools` populated.

        """
        # `self._config.tools` is populated by AgentConfig based on Hydra config.
        if self._config.tools:
            logger.debug(f"Agent {self.agent_name}: Loading tools: {list(self._config.tools.keys())}")

            # Instantiate tools here if they are OmegaConf objects
            _tool_objects = hydra.utils.instantiate(self._config.tools)

            # Uses utility function to convert tool configurations into Autogen-compatible tool formats.
            self._tools = create_tool_functions(_tool_objects)
        else:
            logger.debug(f"Agent '{self.agent_name}': No tools configured.")
            self._tools = []
        return self

    def get_available_tools(self) -> list["Tool"]:
        """Get list of tools this agent can respond to.
        
        Returns the configured tools from self._tools.
        
        Returns:
            list[Tool]: List of configured tools.

        """
        return self._tools

    def get_display_name(self) -> str:
        """Get the display name for this LLM agent, including model information.
        
        Extends the base agent display name to include model tag for UI consistency.
        
        Returns:
            str: Display name with model tag appended

        """
        base_name = self._config.get_display_name()
        model_tag = self._get_model_tag()
        if model_tag:
            return f"{base_name} [{model_tag}]"
        return base_name

    def _get_model_tag(self) -> str:
        """Extract a short tag from the model name for display purposes.
        
        Returns:
            str: Short model identifier (e.g., 'GPT4', 'SONN', 'OPUS')

        """
        if not self.parameters["model"]:
            return ""

        model_lower = self.parameters["model"].lower()

        # Common model patterns
        if "gpt-4" in model_lower:
            return "GPT4"
        if "gpt-3.5" in model_lower:
            return "GPT3"
        if "sonnet" in model_lower:
            return "SONN"
        if "opus" in model_lower:
            return "OPUS"
        if "haiku" in model_lower:
            return "HAIK"
        if "claude" in model_lower:
            return "CLDE"
        if "gemini" in model_lower:
            return "GEMN"
        if "llama" in model_lower:
            return "LLMA"
        return ""

    async def _fill_template(
        self,
        task_params: dict[str, Any],
        inputs: dict[str, Any],
        context: list[LLMMessage] | None = None,  # Made context optional
        records: list[Record] | None = None,   # Made records optional
    ) -> list[LLMMessage]:
        """Renders the agent's prompt template (e.g., Jinja2/Prompty) with provided data.

        This method constructs the list of messages to be sent to the LLM.
        It determines the correct prompt template to use based on `self.parameters`,
        `task_params` (from `AgentInput.parameters`), or `inputs` (from `AgentInput.inputs`).
        It then renders this template, injecting `inputs` as direct template
        variables, and `context` and `records` into specific placeholders within
        the template structure (if using Prompty format).

        Args:
            task_params: Parameters specific to the current task/request, typically
                from `AgentInput.parameters`. These can override agent defaults.
            inputs: Input data provided in the `AgentInput.inputs` dictionary. These
                are made available as variables during template rendering.
            context: Optional. Conversation history as a list of `LLMMessage` objects.
                Used to fill context placeholders in the prompt. Defaults to empty list.
            records: Optional. List of `Record` objects associated with the current task.
                Used to fill record placeholders in the prompt. Defaults to empty list.

        Returns:
            list[LLMMessage]: A list of `LLMMessage` objects (e.g., `SystemMessage`,
            `UserMessage`, `AssistantMessage`) ready to be sent to the LLM.

        Raises:
            ProcessingError: If no prompt template name is defined in the configuration,
                if the template parsing fails, or if `fail_on_unfilled_parameters`
                is True and required template variables are missing from `inputs`.

        """
        # Ensure context and records are lists if None
        current_context = context if context is not None else []
        current_records = records if records is not None else []

        # Template name must be provided in parameters - no fallback chain allowed
        if "template" not in self.parameters:
            raise ProcessingError(f"Agent '{self.agent_id}': 'template' is required in agent parameters.")

        template_name = self.parameters["template"]
        if not template_name or not isinstance(template_name, str):
            raise ProcessingError(f"Agent '{self.agent_id}': 'template' parameter must be a non-empty string.")
        logger.debug(f"Agent '{self.agent_name}': Using prompt template '{template_name}'.")

        combined_params = {**(self.parameters if self.parameters is not None else {}), **(task_params if task_params is not None else {})}

        # Check if prompt is already in context to avoid duplication
        # If the last message in context has the same content as the prompt, don't include it again
        filtered_inputs = inputs.copy() if inputs else {}
        if current_context and "prompt" in filtered_inputs:
            last_context_msg = current_context[-1]
            if isinstance(last_context_msg, UserMessage) and last_context_msg.content == filtered_inputs["prompt"]:
                logger.debug(f"Agent '{self.agent_name}': Removing duplicate prompt from inputs (already in context)")
                del filtered_inputs["prompt"]

        rendered_template_str, unfilled_vars = load_template(
            template=template_name,
            parameters=combined_params,
            untrusted_inputs=filtered_inputs,
        )

        try:
            llm_messages: list[LLMMessage] = make_messages(
                local_template=rendered_template_str,
                context=current_context,
                records=current_records,
            )
        except Exception as e:
            logger.error(f"Agent '{self.agent_name}': Failed to create messages from Prompty structure for template '{template_name}': {e!s}")
            raise ProcessingError(f"Failed to create messages from template '{template_name}' for agent '{self.agent_id}'") from e

        # Identify missing variables, excluding 'records' and 'context' which are handled by make_messages
        missing_vars_in_template = set(unfilled_vars) - {"records", "context"}
        if missing_vars_in_template:
            err_msg = (
                f"Agent '{self.agent_id}' template '{template_name}' has unfilled parameters: "
                f"{', '.join(sorted(list(missing_vars_in_template)))}"
            )
            if self.fail_on_unfilled_parameters:
                logger.error(err_msg)
                raise ProcessingError(err_msg)
            logger.warning(f"{err_msg}. Proceeding as fail_on_unfilled_parameters is False.")

        logger.debug(f"Agent '{self.agent_name}': Template '{template_name}' rendered into {len(llm_messages)} messages for LLM.")
        return llm_messages

    async def _process(self, *, message: AgentInput,
        cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput:
        """Core processing logic: fills template, calls LLM, makes AgentOutput.

        Args:
            message: The `AgentInput` message containing data, context, and
                parameters for this processing step.
            cancellation_token: Optional. A token to signal cancellation of the
                LLM call or other async operations.
            **kwargs: Additional keyword arguments (currently not explicitly used but
                provides flexibility for future extensions or subclass overrides).

        Returns:
            AgentOutput: An `AgentOutput` message. The `outputs` attribute will
            contain the processed LLM response (either a string, a parsed Pydantic
            model if `_output_model` was used, or a generic JSON structure).
            The `metadata` attribute will include information from the LLM call
            (e.g., token usage) and details about the agent.

        Raises:
            ProcessingError: If template filling fails and `fail_on_unfilled_parameters`
                is True, or if the LLM call itself fails after retries, or if
                parsing/validation of the LLM response fails critically.
            FatalError: If the LLM client fails to initialize (caught in `init_model`).

        """
        logger.debug(f"Agent '{self.agent_name}' starting _process for message_id: {getattr(message, 'message_id', 'N/A')}.")

        try:
            llm_messages_to_send = await self._fill_template(
                task_params=message.parameters if message.parameters is not None else {},
                inputs=message.inputs if message.inputs is not None else {},
                context=message.context,
                records=message.records,
            )
        except ProcessingError as template_fill_error:  # Catch specific ProcessingError from _fill_template
            msg = f"Critical error during prompt template processing: {template_fill_error!s}"
            logger.error(f"Agent '{self.agent_id}': {msg}", exc_info=False)
            error_event = ErrorEvent(source=self.agent_id, content=msg)
            return AgentOutput(agent_id=self.agent_id, error=[error_event])

        except Exception as e:  # Catch any other unexpected error during templating
            msg = f"Unexpected template error: {e!s}"
            logger.error(f"Agent '{self.agent_id}': {msg}", exc_info=False)
            error_event = ErrorEvent(source=self.agent_id, content=msg)
            return AgentOutput(agent_id=self.agent_id, error=[error_event])

        tool_names = [getattr(tool, "name", str(tool)) for tool in self._tools]
        logger.info(
            f"Agent '{self.agent_name}': Sending {len(llm_messages_to_send)} messages to LLM '{self.parameters['model']}'. "
            f"Configured tools ({len(self._tools)}): {tool_names}",
        )

        # Call the LLM through our helper method
        chat_result = await self._call_llm(
            messages=llm_messages_to_send,
            tools=self._tools,
            schema=self._output_model,
            cancellation_token=cancellation_token,
        )

        llm_messages_to_send.append(
            AssistantMessage(content=chat_result.content, thought=getattr(chat_result, "thought", None), source=self.agent_id)
        )
        logger.info(
            f"Agent {self.agent_name}: Received response from model '{self.parameters['model']}'. Finish reason: {chat_result.finish_reason}",
        )

        # Extract the final output based on whether we have structured output
        if self._output_model and isinstance(chat_result, ModelOutput):
            final_output = chat_result.parsed_object
        else:
            final_output = chat_result.content

        # Prepare metadata for AgentOutput
        output_metadata = {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "agent_model": self.parameters["model"],
            "finish_reason": chat_result.finish_reason,
        }

        # Only add usage if present
        if chat_result.usage:
            output_metadata["usage"] = chat_result.usage

        logger.info(f"Agent '{self.agent_name}' completed _process. Output type: {type(final_output).__name__}")
        return AgentOutput(agent_id=self.agent_id, outputs=final_output, metadata=output_metadata)

    async def _call_llm(
        self,
        messages: list[LLMMessage],
        tools: list[Tool],
        schema: type[pydantic.BaseModel] | None,
        cancellation_token: CancellationToken | None,
    ) -> CreateResult | ModelOutput:
        """Helper method to call the LLM with proper error handling.

        This method can be overridden by subclasses that need special LLM handling.

        Args:
            messages: The messages to send to the LLM
            tools: Available tools for the LLM to call
            schema: Optional Pydantic schema for structured output
            cancellation_token: Optional cancellation token

        Returns:
            CreateResult | ModelOutput: The LLM response, potentially with parsed object

        Raises:
            ProcessingError: If the LLM call fails

        """
        # Get the appropriate AutoGenWrapper instance from the global `bm.llms` manager.
        model_client = bm.llms.get_autogen_chat_client(self.parameters["model"])

        logger.debug(f"Agent {self.agent_name}: Calling LLM with schema: {schema} (type: {type(schema)})")
        try:
            chat_result = await model_client.call_chat(
                messages=messages,
                tools_list=tools,
                cancellation_token=cancellation_token,
                schema=schema,
            )
        except Exception as llm_error:
            msg = f"Agent {self.agent_id}: Error during LLM call to '{self.parameters['model']}': {llm_error}"
            logger.error(msg)
            raise ProcessingError(msg) from llm_error

        return chat_result

    async def _sequence(self) -> AsyncGenerator[Any, None]:
        """Not implemented for LLMAgent.

        LLMAgent does not use the sequencing functionality from the base Agent class.
        This method is a placeholder to satisfy the abstract base class requirement.

        Yields:
            Never yields anything.

        """
        # LLMAgent doesn't typically use _sequence, but must implement it
        yield  # This makes it a generator but doesn't yield any actual values
