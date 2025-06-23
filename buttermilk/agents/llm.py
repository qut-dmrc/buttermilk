"""Defines `LLMAgent`, a base class for Buttermilk agents that interact with Language Models.

This module provides the core `LLMAgent` class, which extends the base `Agent`
to include functionalities specific to LLM interactions. These include:
-   Initializing an LLM client based on configuration.
-   Loading and managing tools (functions) that the LLM can call.
-   Rendering prompt templates (supporting Jinja2 and Prompty formats).
-   Executing calls to the LLM API.
-   Parsing the LLM's response, with optional validation against a Pydantic schema
    if structured output is expected.
-   Constructing standardized `AgentTrace` messages to record the interaction.
"""

from typing import Any, Self  # For type hinting, Type for Pydantic model hinting

import pydantic  # Pydantic core

# Import Autogen components used for type hints and LLM interaction
from autogen_core import CancellationToken, FunctionCall  # Autogen core types
from autogen_core.models import (  # Autogen message and model types
    AssistantMessage,
)
from autogen_core.tools import Tool  # Autogen tool types
from pydantic import BaseModel, Field, PrivateAttr  # Pydantic components

from buttermilk import buttermilk as bm  # Global Buttermilk instance for framework access

# Buttermilk core imports
from buttermilk._core.agent import Agent, UserMessage  # Base agent and message types
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    ErrorEvent,
    LLMMessage,  # Type hint for message lists
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import CreateResult, ModelOutput  # LLM client wrapper and results
from buttermilk._core.log import logger
from buttermilk._core.types import Record  # Data record type
from buttermilk.utils._tools import create_tool_functions  # Tool handling utility
from buttermilk.utils.json_parser import ChatParser  # JSON parsing utility
from buttermilk.utils.templating import (
    SystemMessage,
    _parse_prompty,  # Prompty parsing utility
    load_template,  # Template loading utility
    make_messages,  # Message list creation utility
)


class LLMAgent(Agent):
    """Base class for Buttermilk agents that interact with Large Language Models (LLMs).

    This class provides a foundational structure for agents that need to communicate
    with LLMs. It handles common tasks such as initializing the LLM client,
    loading and preparing tools for the LLM to use, rendering prompt templates
    (supporting Jinja2 and Prompty formats through utilities), making calls to
    the LLM API (via an `AutoGenWrapper`), and parsing the LLM's response.
    It can also validate structured JSON output from the LLM against a specified
    Pydantic model.

    Subclasses typically need to:
    -   Define the `_output_model` class attribute if they expect structured output
        from the LLM that should be parsed into a specific Pydantic model.
    -   Implement specific message handlers (e.g., methods decorated with
        `@buttermilk_handler`) that prepare an `AgentInput` and then call the
        `self.invoke()` method (which in turn calls `self._process()`) to interact
        with the LLM.
    -   Provide agent-specific configuration (like the LLM model name, prompt
        template name, operational parameters, and tool definitions) through
        Hydra configuration files, which are mapped to `AgentConfig`.

    Attributes:
        fail_on_unfilled_parameters: If True, raise error if template vars aren't filled.
        _tools_list: List of tools available to the LLM.
        _model: Name of the LLM model to use.
        _output_model: Optional Pydantic model to validate/parse the LLM output against.
        _json_parser: Utility for parsing JSON strings.

    """

    fail_on_unfilled_parameters: bool = Field(
        default=True,
        description="If True, raises ProcessingError if prompt template parameters are missing. Otherwise, warns and proceeds.",
    )

    _tools_list: list[Tool] = PrivateAttr(default_factory=list)
    _model: str = PrivateAttr(default="")
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    # Subclasses should override this if they expect specific structured output
    _output_model: type[BaseModel] | None = PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        """Stores the LLM model client name from agent parameters."""
        # Model name must be provided - no fallbacks allowed
        if "model" not in self.parameters:
            raise ValueError(f"Agent {self.agent_id}: 'model' is required in agent parameters.")
        
        self._model = self.parameters["model"]
        if not self._model:
            raise ValueError(f"Agent {self.agent_id}: 'model' parameter cannot be empty.")

        return self

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        """Loads and prepares tools defined in the agent configuration.

        This Pydantic validator runs after the agent model is created.
        It checks `self.tools` (an `AgentConfig` field, typically populated from
        Hydra configuration) and uses `create_tool_functions` to convert these
        tool definitions into a list of Autogen-compatible tool objects
        (e.g., `FunctionTool`). The result is stored in `self._tools_list`.

        Returns:
            Self: The agent instance with `_tools_list` populated.

        """
        # `self.tools` is populated by AgentConfig based on Hydra config.
        if self.tools:
            logger.debug(f"Agent {self.agent_name}: Loading tools: {list(self.tools.keys())}")
            # Uses utility function to convert tool configurations into Autogen-compatible tool formats.
            self._tools_list = create_tool_functions(self.tools)
        else:
            logger.debug(f"Agent '{self.agent_name}': No tools configured.")
            self._tools_list = []
        return self

    def get_display_name(self) -> str:
        """Get the display name for this LLM agent, including model information.
        
        Extends the base agent display name to include model tag for UI consistency.
        
        Returns:
            str: The display name with model tag (e.g., "judge abc123[GPT4]")
        """
        model_tag = self._get_model_tag()
        if model_tag:
            return f"{self.agent_name}[{model_tag}]"
        return self.agent_name

    def _get_model_tag(self) -> str:
        """Extract model identifier for display purposes.
        
        Returns:
            str: Short model identifier (e.g., "GPT4", "SNNT", "GEMN")
        """
        if not self._model:
            return ""

        model_lower = self._model.lower()
        if "gpt-4" in model_lower or "gpt4" in model_lower:
            return "GPT4"
        elif "gpt-3" in model_lower:
            return "GPT3"
        elif "o3" in model_lower:
            return "O3"
        elif "sonnet" in model_lower:
            return "SNNT"
        elif "opus" in model_lower:
            return "OPUS"
        elif "haiku" in model_lower:
            return "HAIK"
        elif "claude" in model_lower:
            return "CLDE"
        elif "gemini" in model_lower:
            return "GEMN"
        elif "llama" in model_lower:
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

        rendered_template_str, unfilled_vars = load_template(
            template=template_name,
            parameters=combined_params,
            untrusted_inputs=inputs,
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

    # This is the primary method subclasses should override.
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
        if not isinstance(message, AgentInput):  # Should be guaranteed by type hint, but defensive
            raise ProcessingError(f"Agent '{self.agent_id}': _process called with non-AgentInput message type: {type(message)}")

        try:
            llm_messages_to_send = await self._fill_template(
                task_params=message.parameters if message.parameters is not None else {},
                inputs=message.inputs if message.inputs is not None else {},
                context=message.context,
                records=message.records,
            )
        except ProcessingError as template_fill_error:  # Catch specific ProcessingError from _fill_template
            logger.error(f"Agent '{self.agent_id}': Critical error during prompt template processing: {template_fill_error!s}")
            error_event = ErrorEvent(source=self.agent_id, content=str(template_fill_error))
            return AgentOutput(agent_id=self.agent_id, metadata={"error": True, "error_type": "TemplateError"}, outputs=error_event)
        except Exception as e:  # Catch any other unexpected error during templating
            logger.error(f"Agent '{self.agent_id}': Unexpected critical error during template processing: {e!s}", exc_info=True)
            error_event = ErrorEvent(source=self.agent_id, content=f"Unexpected template error: {e!s}")
            return AgentOutput(agent_id=self.agent_id, metadata={"error": True, "error_type": "UnexpectedTemplateError"}, outputs=error_event)

        tool_names = [getattr(tool, 'name', str(tool)) for tool in self._tools_list]
        logger.debug(
            f"Agent '{self.agent_name}': Sending {len(llm_messages_to_send)} messages to LLM '{self._model}'. "
            f"Configured tools ({len(self._tools_list)}): {tool_names}"
        )
        try:
            # Get the appropriate AutoGenWrapper instance from the global `bm.llms` manager.
            model_client = bm.llms.get_autogen_chat_client(self._model)

            # Debug the schema parameter
            logger.debug(f"Agent {self.agent_name}: About to call LLM with schema: {self._output_model} (type: {type(self._output_model)})")

            chat_result: CreateResult = await model_client.call_chat(
                messages=llm_messages_to_send,
                tools_list=self._tools_list,
                cancellation_token=cancellation_token,
                schema=self._output_model,  # Pass expected Pydantic schema for structured output
            )
            llm_messages_to_send.append(AssistantMessage(content=chat_result.content, thought=chat_result.thought, source=self.agent_id))
            logger.debug(f"Agent {self.agent_name}: Received response from model '{self._model}'. Finish reason: {chat_result.finish_reason}")
        except Exception as llm_error:
            # Catch errors during the actual LLM call
            msg = f"Agent {self.agent_id}: Error during LLM call to '{self._model}': {llm_error}"
            raise ProcessingError(msg) from llm_error

        # 3. Parse the LLM response and create an AgentOutput
        parsed_object = None
        if schema := self._output_model:
            if isinstance(chat_result, ModelOutput) and isinstance(chat_result.parsed_object, schema):
                # If client already parsed into the correct schema object
                parsed_object = chat_result.parsed_object
            elif isinstance(chat_result.content, str):
                # If content is a string, try to parse/validate it against the schema
                logger.debug(f"Agent {self.agent_name}: Attempting to parse LLM content into schema {schema.__name__}.")
                try:
                    # Use Pydantic's model_validate_json for robust parsing from JSON string.
                    parsed_object = schema.model_validate_json(chat_result.content)
                    logger.debug(f"Agent {self.agent_name}: Successfully parsed content into schema {schema.__name__}.")
                except Exception as e:
                    # Log schema validation/parsing error
                    parse_error = f"Error parsing LLM response into {schema.__name__}: {e}."
                    logger.debug(f"Agent {self.agent_name}: {parse_error}")
            else:
                # Content is not string and not pre-parsed object, schema validation fails.
                parse_error = f"LLM response content type ({type(chat_result.content)}) is incompatible with schema {schema.__name__}."
                raise ProcessingError(f"Agent {self.agent_name}: {parse_error}")

        if parsed_object is None:
            if isinstance(chat_result.content, str):
                # If no schema or schema validation failed, try parsing as generic JSON
                logger.debug(f"Agent {self.agent_name}: Attempting generic JSON parsing of LLM content.")
                try:
                    parsed_object = self._json_parser.parse(chat_result.content)
                    logger.debug(f"Agent {self.agent_name}: Successfully parsed content as generic JSON.")
                except ProcessingError:
                    # If generic JSON parsing also fails, treat as a chat message
                    if isinstance(chat_result, (AssistantMessage, SystemMessage, UserMessage)):
                        # If the content is already an AssistantMessage, use it directly
                        parsed_object = chat_result
                    else:
                        parsed_object = AssistantMessage(content=chat_result.content,
                                                   thought=chat_result.thought,
                                                   source=self.agent_id)
            else:
                # Content is not parseable
                raise ProcessingError(f"Agent {self.agent_id} was unable to parse LLM response content of type: ({type(chat_result.content)}).")

        # Add agent role/name for context in logs/outputs
        metadata = {k: v for k, v in vars(chat_result).items() if k not in {"content"}}
        metadata.update({"role": self.role, "name": self.agent_name})

        # Return an AgentOutput
        response = AgentOutput(agent_id=self.agent_id,
            metadata=metadata,
            outputs=parsed_object,
        )

        logger.debug(f"Agent {self.agent_name} finished _process.")
        return response

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset any LLM-specific state for the agent.

        Calls the `super().on_reset()` from the base `Agent` class to clear
        common state like records, model context, and data. Subclasses of
        `LLMAgent` can override this to add further reset logic specific to
        their needs (e.g., clearing LLM-specific caches or stateful tool instances).

        Args:
            cancellation_token: An optional `CancellationToken` to signal if
                the reset operation should be aborted.

        """
        await super().on_reset(cancellation_token=cancellation_token)
        # Add any LLMAgent-specific state reset here if needed in the future.
        # For example, if LLMAgent maintained a specific type of conversation history
        # or cached LLM responses that need clearing beyond what base Agent.on_reset does.
        logger.debug(f"LLMAgent '{self.agent_name}' state reset (invoked super().on_reset).")
