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

from typing import Any, Self, Type  # For type hinting, Type for Pydantic model hinting

import pydantic # Pydantic core
import weave # For tracing capabilities

# Import Autogen components used for type hints and LLM interaction
from autogen_core import CancellationToken, FunctionCall # Autogen core types
from autogen_core.models import ( # Autogen message and model types
    AssistantMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema # Autogen tool types
from pydantic import BaseModel, Field, PrivateAttr # Pydantic components

from buttermilk import buttermilk as bm  # Global Buttermilk instance for framework access

# Buttermilk core imports
from buttermilk._core.agent import Agent  # Base Buttermilk agent class
from buttermilk._core.contract import ( # Buttermilk message contracts
    AgentInput,
    AgentOutput,
    AgentTrace,
    ErrorEvent, # For error reporting
    LLMMessage,  # Union type for LLM messages
)
from buttermilk._core.exceptions import FatalError, ProcessingError # Custom exceptions
from buttermilk._core.llms import AutoGenWrapper, CreateResult, ModelOutput  # LLM client wrapper and result types
from buttermilk._core.log import logger # Centralized logger
from buttermilk._core.types import Record  # Core Buttermilk Record data type
from buttermilk.utils._tools import create_tool_functions  # Tool handling utility
from buttermilk.utils.json_parser import ChatParser  # JSON parsing utility
from buttermilk.utils.templating import ( # Prompt templating utilities
    _parse_prompty,
    load_template,
    make_messages,
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
        fail_on_unfilled_parameters (bool): If True (default), the agent will
            raise a `ProcessingError` during prompt template rendering if any
            expected parameters are missing. If False, it will log a warning
            and proceed with potentially incomplete prompts.
        _tools_list (list[FunctionCall | Tool | ToolSchema | FunctionTool]):
            A list of Autogen-compatible tools available to the LLM for this agent.
            Populated by `_load_tools` based on agent configuration.
        _model (str): The name of the LLM model to be used (e.g., "gpt-4", "claude-3").
            Initialized by `init_model` from agent parameters.
        _model_client (AutoGenWrapper): An instance of `AutoGenWrapper` that wraps
            the actual LLM client, providing retry and rate-limiting logic.
            Initialized by `init_model`.
        _output_model (Type[BaseModel] | None): An optional Pydantic model class.
            If provided, the LLM's JSON output will be parsed and validated against
            this schema. Subclasses should set this attribute at the class level.
        _json_parser (ChatParser): A utility for parsing JSON strings, potentially
            extracting JSON from within larger text blocks.
    """

    fail_on_unfilled_parameters: bool = Field(
        default=True, 
        description="If True, raises ProcessingError if prompt template parameters are missing. Otherwise, warns and proceeds."
    )

    _tools_list: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(default_factory=list)
    _model: str = PrivateAttr(default="")
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: AutoGenWrapper = PrivateAttr(default=None) # type: ignore[assignment] # Initialized in validator
    
    # Subclasses should override this class attribute if they expect specific structured output
    _output_model: Type[BaseModel] | None = None # Changed PrivateAttr to class attribute

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        """Initializes the LLM model client based on agent parameters.

        This Pydantic validator runs after the agent model is first created.
        It retrieves the LLM model name from `self.parameters["model"]` and uses
        the global Buttermilk instance (`bm.llms`) to get an initialized
        `AutoGenWrapper` for that model.

        Returns:
            Self: The agent instance with `_model` and `_model_client` initialized.

        Raises:
            ValueError: If the "model" parameter is not found in `self.parameters`.
            FatalError: If the LLM client fails to initialize (e.g., model not
                configured in `bm.llms` or API key issues).
        """
        model_name_param = self.parameters.get("model")
        if not model_name_param or not isinstance(model_name_param, str):
            raise ValueError(f"Agent '{self.agent_id}': LLM model name ('parameters.model') must be a non-empty string.")
        self._model = model_name_param

        logger.debug(f"Agent '{self.agent_name}': Initializing model client for '{self._model}'.")
        try:
            self._model_client = bm.llms.get_autogen_chat_client(self._model)
        except Exception as e:
            err_msg = f"Agent '{self.agent_id}': Failed to initialize model client for '{self._model}'. Error: {e!s}"
            logger.error(err_msg)
            raise FatalError(err_msg) from e
        return self

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        """Loads and prepares tools defined in the agent's configuration.

        This Pydantic validator runs after the agent model is created.
        It checks `self.tools` (an `AgentConfig` field, typically populated from
        Hydra configuration) and uses `create_tool_functions` to convert these
        tool definitions into a list of Autogen-compatible tool objects
        (e.g., `FunctionTool`). The result is stored in `self._tools_list`.

        Returns:
            Self: The agent instance with `_tools_list` populated.
        """
        if self.tools: # self.tools is from AgentConfig
            tool_names = [tool_cfg.tool_obj for tool_cfg in self.tools if hasattr(tool_cfg, 'tool_obj')] # Assuming tool_obj is the name
            logger.debug(f"Agent '{self.agent_name}': Loading tools: {tool_names}")
            try:
                self._tools_list = create_tool_functions(self.tools)
            except Exception as e:
                logger.error(f"Agent '{self.agent_name}': Failed to load/create tools: {e!s}")
                self._tools_list = [] # Default to empty on error
        else:
            logger.debug(f"Agent '{self.agent_name}': No tools configured.")
            self._tools_list = []
        return self

    async def _fill_template(
        self,
        task_params: dict[str, Any],
        inputs: dict[str, Any],
        context: list[LLMMessage] | None = None, # Made context optional
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
        current_context = context or []
        current_records = records or []

        template_name = self.parameters.get("template", task_params.get("template", inputs.get("template")))
        if not template_name or not isinstance(template_name, str):
            raise ProcessingError(f"Agent '{self.agent_id}': No valid template name provided in parameters or inputs.")
        logger.debug(f"Agent '{self.agent_name}': Using prompt template '{template_name}'.")

        combined_params = {**(self.parameters or {}), **(task_params or {})}

        rendered_template_str, unfilled_vars = load_template(
            template=template_name,
            parameters=combined_params,
            untrusted_inputs=inputs,
        )

        try:
            prompty_structure = _parse_prompty(rendered_template_str)
        except Exception as e:
            logger.error(f"Agent '{self.agent_name}': Failed to parse rendered template '{template_name}' as Prompty: {e!s}")
            raise ProcessingError(f"Failed to parse template '{template_name}' for agent '{self.agent_id}'") from e

        try:
            llm_messages: list[LLMMessage] = make_messages(
                local_template=prompty_structure, context=current_context, records=current_records
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

    def make_trace(
        self,
        chat_result: CreateResult,
        inputs: AgentInput,
        messages: list[LLMMessage] | None = None, # Made optional
        schema: Type[BaseModel] | None = None, # Use Type for class reference
    ) -> AgentTrace:
        """Constructs a standardized `AgentTrace` message from the LLM interaction results.

        This method takes the raw result from the LLM client (`chat_result`),
        the original `AgentInput` that triggered the call, the list of messages
        sent to the LLM, and an optional Pydantic schema (`self._output_model`).

        It performs the following steps:
        1.  Initializes an `AgentTrace` object with basic information (agent ID,
            session ID, agent config, original inputs, LLM messages, and metadata
            from `chat_result`).
        2.  Adds a Weave tracing link to `output.tracing_link` if available.
        3.  If a `schema` (typically `self._output_model`) is provided:
            a.  If `chat_result` is a `ModelOutput` and its `object` attribute
                is already an instance of the schema, that object is used.
            b.  If `chat_result.content` is a string, it attempts to parse and
                validate this string as JSON against the `schema`.
            c.  If parsing/validation fails or content is incompatible, raises `ProcessingError`.
        4.  If a `parsed_object` (from schema validation) exists, it's set as `output.outputs`.
        5.  Otherwise (no schema or schema validation failed but didn't raise to stop):
            a.  If `chat_result.content` is a string, it attempts to parse it as
                generic JSON using `self._json_parser`.
            b.  If generic JSON parsing fails, or if content was not a string but
                is not None, raises `ProcessingError`.
            c.  If `chat_result.content` is None, raises `ProcessingError`.

        Args:
            chat_result: The `CreateResult` object returned by the LLM client
                wrapper (`AutoGenWrapper.call_chat`).
            inputs: The original `AgentInput` that initiated this LLM call.
            messages: Optional. The list of `LLMMessage` objects that were actually
                sent to the LLM. If None, defaults to an empty list.
            schema: Optional. The Pydantic model class against which the LLM's output
                should be validated. Typically `self._output_model`.

        Returns:
            AgentTrace: An `AgentTrace` object populated with details of the LLM
            interaction and the processed output.

        Raises:
            ProcessingError: If parsing or validation of the LLM's content fails
                based on the presence/absence of a schema and the content type.
        """
        llm_messages_sent = messages or []
        output = AgentTrace(
            agent_id=self.agent_id, session_id=self.session_id,
            agent_info=self._cfg,
            inputs=inputs,
            messages=llm_messages_sent,
            metadata=chat_result.model_dump(exclude={"content", "object"}, exclude_none=True), # Exclude none from metadata
        )

        parsed_llm_output: Any = None # Holds the successfully parsed output
        
        # Attempt to get Weave tracing link
        try:
            weave_call = weave.get_current_call()
            if weave_call and hasattr(weave_call, "ui_url") and weave_call.ui_url:
                output.tracing_link = weave_call.ui_url
                output.call_id = weave_call.id or output.call_id # Update call_id from Weave if available
            # else: No active Weave call or UI URL not available.
        except Exception as e: # Catch errors if Weave is not configured or fails
            logger.debug(f"Agent '{self.agent_name}': Could not get Weave call context/UI URL: {e!s}")


        # 1. Handle structured output validation if a schema is defined for this agent
        if schema:
            if isinstance(chat_result, ModelOutput) and chat_result.object and isinstance(chat_result.object, schema):
                parsed_llm_output = chat_result.object # Already parsed by AutoGenWrapper/client
                logger.debug(f"Agent '{self.agent_name}': Used pre-parsed object of schema '{schema.__name__}'.")
            elif isinstance(chat_result.content, str) and chat_result.content.strip():
                logger.debug(f"Agent '{self.agent_name}': Attempting to parse LLM string content into schema '{schema.__name__}'.")
                try:
                    parsed_llm_output = schema.model_validate_json(chat_result.content)
                    logger.debug(f"Agent '{self.agent_name}': Successfully parsed content into schema '{schema.__name__}'.")
                except pydantic.ValidationError as e:
                    err_msg = f"LLM response failed Pydantic validation against schema '{schema.__name__}'. Errors: {e.errors()}"
                    logger.error(f"Agent '{self.agent_id}': {err_msg}\nRaw content: {chat_result.content[:500]}...")
                    raise ProcessingError(f"Agent '{self.agent_id}': {err_msg}") from e
                except json.JSONDecodeError as e: # If content is not valid JSON for model_validate_json
                    err_msg = f"LLM response is not valid JSON, cannot parse into schema '{schema.__name__}'. Error: {e!s}"
                    logger.error(f"Agent '{self.agent_id}': {err_msg}\nRaw content: {chat_result.content[:500]}...")
                    raise ProcessingError(f"Agent '{self.agent_id}': {err_msg}") from e
            elif chat_result.content is None or (isinstance(chat_result.content, str) and not chat_result.content.strip()):
                 err_msg = f"LLM response content is None or empty, cannot parse into schema '{schema.__name__}'."
                 logger.error(f"Agent '{self.agent_id}': {err_msg}")
                 raise ProcessingError(f"Agent '{self.agent_id}': {err_msg}")
            else: # Content is not a string and not a pre-parsed object matching schema
                err_msg = (f"LLM response content type ('{type(chat_result.content).__name__}') is incompatible "
                           f"with schema '{schema.__name__}' for direct parsing.")
                logger.error(f"Agent '{self.agent_id}': {err_msg}")
                raise ProcessingError(f"Agent '{self.agent_id}': {err_msg}")
        
        # 2. If no schema, or if schema parsing was successful (parsed_llm_output is set)
        if parsed_llm_output is not None:
            output.outputs = parsed_llm_output
        elif isinstance(chat_result.content, str) and chat_result.content.strip():
            # No schema was provided, or schema parsing somehow didn't set parsed_llm_output but didn't error.
            # Attempt generic JSON parsing if it's a string.
            logger.debug(f"Agent '{self.agent_name}': No schema or schema parsing did not populate output. Attempting generic JSON parsing of LLM string content.")
            try:
                output.outputs = self._json_parser.parse(chat_result.content) # ChatParser might extract JSON from text
                logger.debug(f"Agent '{self.agent_name}': Successfully parsed content as generic JSON/extracted value.")
            except Exception as json_e: # If ChatParser also fails
                logger.warning(f"Agent '{self.agent_id}': Failed to parse LLM response as generic JSON (error: {json_e!s}). Storing raw string content.")
                output.outputs = chat_result.content # Store raw string content as fallback
        elif chat_result.content is not None:
            # Content is not None, not a string, and no schema was matched. Store as is.
            logger.debug(f"Agent '{self.agent_name}': LLM response content is not a string ({type(chat_result.content).__name__}) and no schema applied. Storing raw content.")
            output.outputs = chat_result.content
        else: # Content is None
            logger.warning(f"Agent '{self.agent_id}': LLM response content is None. Output will be None.")
            output.outputs = None # Explicitly set to None

        return output

    async def _process(
        self, 
        *, 
        message: AgentInput,
        cancellation_token: CancellationToken | None = None, 
        **kwargs: Any
    ) -> AgentOutput:
        """Core processing logic for the LLM agent.

        This method orchestrates the interaction with the LLM:
        1.  Renders the prompt template using `_fill_template`, providing it with
            data from the input `message` (parameters, inputs, context, records).
        2.  Calls the configured LLM via `self._model_client.call_chat`, passing
            the rendered messages and any configured tools (`self._tools_list`).
            It also passes `self._output_model` to the client, which might use it
            for requesting structured output (e.g., JSON mode with a schema hint).
        3.  Constructs a standardized `AgentTrace` object from the LLM's response
            using `self.make_trace`. This trace includes the original input,
            messages sent to the LLM, parsed outputs, and metadata.
        4.  Wraps the relevant parts of the `AgentTrace` (outputs, metadata) into
            an `AgentOutput` object, which is the standard return type for `_process`.

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
        if not isinstance(message, AgentInput): # Should be guaranteed by type hint, but defensive
            raise ProcessingError(f"Agent '{self.agent_id}': _process called with non-AgentInput message type: {type(message)}")
        
        try:
            llm_messages_to_send = await self._fill_template(
                task_params=message.parameters or {}, # Ensure dict
                inputs=message.inputs or {},       # Ensure dict
                context=message.context,
                records=message.records,
            )
        except ProcessingError as template_fill_error: # Catch specific ProcessingError from _fill_template
            logger.error(f"Agent '{self.agent_id}': Critical error during prompt template processing: {template_fill_error!s}")
            error_event = ErrorEvent(source=self.agent_id, content=str(template_fill_error))
            return AgentOutput(agent_id=self.agent_id, metadata={"error": True, "error_type": "TemplateError"}, outputs=error_event)
        except Exception as e: # Catch any other unexpected error during templating
            logger.error(f"Agent '{self.agent_id}': Unexpected critical error during template processing: {e!s}", exc_info=True)
            error_event = ErrorEvent(source=self.agent_id, content=f"Unexpected template error: {e!s}")
            return AgentOutput(agent_id=self.agent_id, metadata={"error": True, "error_type": "UnexpectedTemplateError"}, outputs=error_event)

        logger.debug(f"Agent '{self.agent_name}': Sending {len(llm_messages_to_send)} messages to LLM '{self._model}'.")
        try:
            llm_api_result: CreateResult = await self._model_client.call_chat(
                messages=llm_messages_to_send,
                tools_list=self._tools_list, # Pass configured tools
                cancellation_token=cancellation_token,
                schema=self._output_model,  # Pass expected Pydantic schema for structured output
            )
            # Append the assistant's response to the list of messages for the trace
            # Ensure content is a string, as LLMMessage expects str for content.
            # If llm_api_result.content is complex (e.g., list of FunctionCall),
            # this might need adjustment or specific handling if tools are used.
            assistant_response_content = llm_api_result.content
            if not isinstance(assistant_response_content, str):
                assistant_response_content = str(assistant_response_content) # Fallback to stringify

            llm_messages_to_send.append(
                AssistantMessage(
                    content=assistant_response_content, 
                    thought=getattr(llm_api_result, 'thought', None), # Include thought if present
                    source=self.agent_id # Source is this agent
                )
            )
            logger.debug(f"Agent '{self.agent_name}': Received response from LLM '{self._model}'. Finish reason: {llm_api_result.finish_reason}")
        except ProcessingError as pe: # Re-raise known processing errors (e.g. from retry wrapper)
            logger.error(f"Agent '{self.agent_id}': ProcessingError during LLM call to '{self._model}': {pe!s}", exc_info=True)
            raise # Keep original stack trace if possible
        except Exception as llm_api_error: # Catch other errors from LLM call
            err_msg = f"Agent '{self.agent_id}': Unhandled error during LLM call to '{self._model}': {llm_api_error!s}"
            logger.error(err_msg, exc_info=True)
            raise ProcessingError(err_msg) from llm_api_error # Wrap in ProcessingError

        # Create the standardized AgentTrace using the result from LLM
        try:
            agent_trace = self.make_trace(
                chat_result=llm_api_result,
                inputs=message,
                messages=llm_messages_to_send, # Pass all messages including assistant's response
                schema=self._output_model,
            )
        except ProcessingError as trace_creation_error: # Catch errors from make_trace (e.g., parsing issues)
            logger.error(f"Agent '{self.agent_id}': Error creating AgentTrace: {trace_creation_error!s}", exc_info=True)
            # Return an AgentOutput with error information
            error_event = ErrorEvent(source=self.agent_id, content=str(trace_creation_error))
            return AgentOutput(
                agent_id=self.agent_id, 
                metadata={"error": True, "error_type": "TraceCreationError"}, 
                outputs=error_event,
                # call_id might be from llm_api_result if available, or default from AgentOutput
                call_id=getattr(llm_api_result, "id", None) 
            )

        # Populate AgentOutput from the AgentTrace
        # The Agent's __call__ method (which invokes _process) will typically wrap this AgentOutput
        # into a full AgentTrace if this _process is called by self.invoke().
        # If _process is called directly, returning AgentOutput is fine.
        final_agent_output = AgentOutput(
            agent_id=self.agent_id,
            metadata=agent_trace.metadata, # Metadata from trace (includes LLM usage, etc.)
            outputs=agent_trace.outputs,   # Parsed/validated outputs from trace
            call_id=agent_trace.call_id,   # Ensure call_id from trace is used
        )
        # If AgentTrace itself had errors from FlowMessage base, they'd be in agent_trace.error
        # This could be copied to final_agent_output.error if AgentOutput also had an error field.
        # For now, assuming errors are handled by raising ProcessingError or within trace.outputs.

        logger.debug(f"Agent '{self.agent_name}' finished _process successfully.")
        return final_agent_output

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Resets any LLM-specific state for the agent.

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
