"""Defines the LLMAgent, a base class for Buttermilk agents that interact with Language Models.
"""

from collections.abc import Callable
from typing import Any, Self  # Added Type for Pydantic model hinting

import pydantic

# Import Autogen components used for type hints and LLM interaction
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema
from pydantic import BaseModel, Field, PrivateAttr

# Buttermilk core imports
from buttermilk._core.agent import Agent  # Base agent and message types
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AgentTrace,
    # TODO: Review necessary contract types for this base class
    ErrorEvent,
    LLMMessage,  # Type hint for message lists
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import AutoGenWrapper, CreateResult, ModelOutput  # LLM client wrapper and results
from buttermilk._core.types import Record  # Data record type
from buttermilk.bm import bm, logger  # Global instance and logger
from buttermilk.utils._tools import create_tool_functions  # Tool handling utility
from buttermilk.utils.json_parser import ChatParser  # JSON parsing utility
from buttermilk.utils.templating import (
    _parse_prompty,  # Prompty parsing utility
    load_template,  # Template loading utility
    make_messages,  # Message list creation utility
)


class LLMAgent(Agent):
    """Base class for Buttermilk agents that interact with Large Language Models (LLMs).

    Handles common tasks such as:
    - Initializing the LLM client based on configuration.
    - Loading and processing tools.
    - Rendering prompt templates (Jinja2/Prompty).
    - Calling the LLM API.
    - Parsing the LLM response, optionally validating against a Pydantic schema.
    - Constructing standardized AgentTrace messages.

    Subclasses typically need to:
    - Define `_output_model` if structured output is expected.
    - Implement specific message handlers (e.g., using `@buttermilk_handler`)
      that call the `_process` method.
    - Provide agent-specific configuration (template, model, parameters, tools) via Hydra.

    Attributes:
        fail_on_unfilled_parameters: If True, raise error if template vars aren't filled.
        _tools_list: List of tools available to the LLM.
        _model: Name of the LLM model to use.
        _model_client: Instance of the LLM client wrapper (AutoGenWrapper).
        _output_model: Optional Pydantic model to validate/parse the LLM output against.
        _json_parser: Utility for parsing JSON strings.

    """

    # Configuration Fields (set via Hydra/Pydantic)
    fail_on_unfilled_parameters: bool = Field(default=True, description="If True, raise ProcessingError if template parameters are missing.")

    # Private Attributes (managed internally)
    _tools_list: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(default_factory=list)
    _model: str = PrivateAttr(default="")  # Populated by init_model validator
    # _name_components is inherited from Agent, used for generating agent ID/name.
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: AutoGenWrapper = PrivateAttr(default=None)  # Populated by init_model validator
    # Subclasses should override this if they expect specific structured output
    _output_model: type[BaseModel] | None = PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        """Initializes the LLM model client based on agent parameters."""
        # Retrieves model name from 'parameters' section of agent config.
        self._model = self.parameters.get("model")
        if not self._model:
            # TODO: Maybe allow model to be defined at a higher level (e.g., flow level)?
            raise ValueError(f"Agent {self.agent_id}: LLM model name must be provided in agent parameters.")

        logger.debug(f"Agent {self.agent_id}: Initializing model client for '{self._model}'.")
        try:
            # Get the appropriate AutoGenWrapper instance from the global `bm.llms` manager.
            self._model_client = bm.llms.get_autogen_chat_client(self._model)
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Failed to initialize model client for '{self._model}': {e}")
            # Raise a more informative error or handle appropriately.
            raise ValueError(f"Failed to get model client for '{self._model}'") from e

        if not hasattr(self, "_model_client") or not self._model_client:
            # This check is likely redundant due to the exception handling above, but belts and suspenders.
            raise ValueError(f"Agent {self.agent_id}: Model client initialization failed for '{self._model}'.")

        return self

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        """Loads and prepares tools defined in the agent configuration."""
        # `self.tools` is likely inherited or populated by AgentConfig based on Hydra config.
        if self.tools:
            logger.debug(f"Agent {self.agent_id}: Loading tools: {list(self.tools.keys())}")
            # Uses utility function to convert tool configurations into Autogen-compatible tool formats.
            self._tools_list = create_tool_functions(self.tools)
        else:
            logger.debug(f"Agent {self.agent_id}: No tools configured.")
            self._tools_list = []
        return self

    async def _fill_template(
        self,
        task_params: dict[str, Any],  # Parameters specific to the current task/request.
        inputs: dict[str, Any],  # Input data provided in the AgentInput message.
        context: list[LLMMessage] = [],  # Conversation history.
        records: list[Record] = [],  # Associated data records.
    ) -> list[LLMMessage]:  # Changed return type hint
        """Renders the agent's prompt template (Jinja2/Prompty) with provided data.

        Args:
            task_params: Parameters from the incoming AgentInput message.
            inputs: 'inputs' dictionary from the AgentInput message.
            context: Conversation history (list of LLMMessage).
            records: List of data records.

        Returns:
            A list of messages (LLMMessage, typically SystemMessage, UserMessage)
            ready to be sent to the LLM.

        Raises:
            ProcessingError: If no template is defined or if `fail_on_unfilled_parameters`
                             is True and template variables are missing.

        """
        # Determine the template name from parameters (task-specific, agent default, or input override).
        template_name = self.parameters.get("template", task_params.get("template", inputs.get("template")))
        if not template_name:
            raise ProcessingError(f"Agent {self.agent_id}: No template name provided in parameters or inputs.")
        logger.debug(f"Agent {self.agent_id}: Using template '{template_name}'.")

        # Combine agent default parameters and task-specific parameters. Task parameters override defaults.
        combined_params = {**self.parameters, **task_params}

        # Render the Jinja2 template. `load_template` handles finding and rendering.
        # It takes parameters needed *by the template loading/finding logic itself*
        # and the `untrusted_inputs` which are directly available for filling placeholders.
        rendered_template, unfilled_vars = load_template(
            template=template_name,
            parameters=combined_params,  # parameters for template source logic (e.g., finding criteria file)
            untrusted_inputs=inputs,  # Vars directly available to Jinja {{ }}
        )

        # Parse the rendered content as a Prompty file to extract message structure.
        try:
            prompty_structure = _parse_prompty(rendered_template)
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Failed to parse rendered template '{template_name}' as Prompty: {e}")
            raise ProcessingError(f"Failed to parse template '{template_name}'") from e

        # Convert the Prompty structure into a list of Autogen message objects (System, User, etc.).
        # `make_messages` injects context and records into appropriate placeholders.
        try:
            messages: list[LLMMessage] = make_messages(local_template=prompty_structure, context=context, records=records)
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Failed to create messages from Prompty structure for template '{template_name}': {e}")
            raise ProcessingError(f"Failed to create messages from template '{template_name}'") from e

        # Check for any variables that were expected by the template but not provided in `inputs`.
        # Ignore 'records' and 'context' as they are handled separately by `make_messages`.
        missing_vars = set(unfilled_vars) - {"records", "context"}
        if missing_vars:
            err = f"Agent {self.agent_id} template '{template_name}' has unfilled parameters: {', '.join(missing_vars)}"
            if self.fail_on_unfilled_parameters:
                logger.error(err)
                raise ProcessingError(err)
            logger.warning(err + ". Proceeding anyway.")

        logger.debug(f"Agent {self.agent_id}: Template '{template_name}' rendered into {len(messages)} messages.")
        return messages

    def make_trace(
        self,
        chat_result: CreateResult,  # Raw result from the LLM client wrapper.
        inputs: AgentInput,  # Original AgentInput that triggered this call.
        messages: list[LLMMessage] = [],  # Messages sent to the LLM.
        schema: type[BaseModel] | None = None,  # Optional Pydantic schema for validation.
    ) -> AgentTrace:
        """Constructs a standardized AgentTrace message from the LLM response.

        Parses the LLM content, attempts validation against the schema if provided,
        and populates the AgentTrace fields.

        Args:
            chat_result: The CreateResult object from `AutoGenWrapper.call_chat`.
            inputs: The original AgentInput that triggered this call.
            messages: The list of messages sent to the LLM.
            prompt: The original prompt text from the AgentInput.
            schema: The Pydantic model to validate the LLM output against (_output_model).

        Returns:
            An AgentTrace message containing the processed results and metadata.

        """
        output = AgentTrace(agent_id=self.agent_id, session_id=self.session_id,
            agent_info=self._cfg,  # Include agent information
            inputs=inputs,
            messages=messages,  # Messages sent to LLM
            # Store LLM usage info (tokens, cost) and other metadata from CreateResult.
            # Exclude raw content and object type which are handled below.
            metadata=chat_result.model_dump(exclude={"content", "object"}),
        )

        parsed_object = None
        parse_error = None

        # 1. Handle structured output validation first (if schema is defined)
        if schema:
            if isinstance(chat_result, ModelOutput) and isinstance(chat_result.object, schema):
                # If client already parsed into the correct schema object
                parsed_object = chat_result.object
            elif isinstance(chat_result.content, str):
                # If content is a string, try to parse/validate it against the schema
                logger.debug(f"Agent {self.agent_id}: Attempting to parse LLM content into schema {schema.__name__}.")
                try:
                    # Use Pydantic's model_validate_json for robust parsing from JSON string.
                    parsed_object = schema.model_validate_json(chat_result.content)
                    logger.debug(f"Agent {self.agent_id}: Successfully parsed content into schema {schema.__name__}.")
                except Exception as e:
                    # Log schema validation/parsing error
                    parse_error = f"Error parsing LLM response into {schema.__name__}: {e}"
                    logger.warning(f"Agent {self.agent_id}: {parse_error}")
                    # Keep the error, we'll add it to output later.
            else:
                # Content is not string and not pre-parsed object, schema validation fails.
                parse_error = f"LLM response content type ({type(chat_result.content)}) is incompatible with schema {schema.__name__}."
                logger.warning(f"Agent {self.agent_id}: {parse_error}")

        # 2. Store the result in output.outputs
        if parsed_object is not None:
            output.outputs = parsed_object  # Store the validated Pydantic object
        elif isinstance(chat_result.content, str):
            # If no schema or schema validation failed, try parsing as generic JSON
            logger.debug(f"Agent {self.agent_id}: Attempting generic JSON parsing of LLM content.")
            try:
                output.outputs = self._json_parser.parse(chat_result.content)
                logger.debug(f"Agent {self.agent_id}: Successfully parsed content as generic JSON.")
            except Exception as json_e:
                # If generic JSON parsing also fails, store raw content
                raw_content_error = f"Failed to parse LLM response as JSON: {json_e}. Storing raw content."
                logger.warning(f"Agent {self.agent_id}: {raw_content_error}")
                output.outputs = chat_result.content  # Store raw string
                # Add both schema error (if any) and JSON parsing error
                if parse_error:
                    output.set_error(parse_error)
                output.set_error(raw_content_error)
        elif chat_result.content is not None:
            # Content is not None, not string - store as is.
            logger.warning(f"Agent {self.agent_id}: LLM response content is not a string ({type(chat_result.content)}). Storing as is.")
            output.outputs = chat_result.content
            if parse_error:  # Add schema error if validation failed
                output.set_error(parse_error)
        else:
            # Content is None
            logger.warning(f"Agent {self.agent_id}: LLM response content is None.")
            output.outputs = None
            if parse_error:  # Add schema error if validation failed
                output.set_error(parse_error)

        return output

    # This is the primary method subclasses should call in their handlers.
    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,  # Callback provided by adapter
        message_callback: Callable | None = None,  # Callback provided by adapter,
        **kwargs) -> AgentOutput:
        """Core processing logic: fills template, calls LLM, makes AgentTrace.

        Args:
            message: The input message containing data, context, parameters.
            cancellation_token: Token to signal cancellation.
            **kwargs: Additional arguments (currently unused but allows flexibility).

        Returns:
            An AgentTrace message with the LLM response and metadata.

        Raises:
            ProcessingError: If template filling fails and `fail_on_unfilled_parameters` is True.
            Exception: If the LLM call itself fails unexpectedly.

        """
        logger.debug(f"Agent {self.agent_id} starting _process.")
        if prompt := getattr(message, "prompt", None):
            if message.parameters.get("prompt"):
                logger.warning(f"Agent {self.agent_id}: Both prompt and parameters.prompt provided in AgentInput. Using AgentInput.prompt.")
            message.parameters["prompt"] = prompt  # Override parameters with prompt from message
        try:
            # 1. Prepare messages for the LLM using the template
            llm_messages = await self._fill_template(
                task_params=message.parameters, inputs=message.inputs, context=message.context, records=message.records,
            )
        except ProcessingError as template_error:
            # Log template errors clearly as they prevent LLM call
            msg = f"Agent {self.agent_id}: Critical error during template processing: {template_error}"
            logger.error(msg)
            # Create an error output
            error_output = ErrorEvent(source=self.agent_id, content=msg)
            # Wrap in AgentOutput
            return AgentOutput(agent_id=self.agent_id,
                metadata={"error": True},
                outputs=error_output,
            )

        # 2. Call the LLM
        logger.debug(f"Agent {self.agent_id}: Sending {len(llm_messages)} messages to model '{self._model}'.")
        try:
            llm_result: CreateResult = await self._model_client.call_chat(
                messages=llm_messages,
                tools_list=self._tools_list,
                cancellation_token=cancellation_token,
                schema=self._output_model,  # Pass expected schema to client if supported
            )
            llm_messages.append(AssistantMessage(content=llm_result.content, thought=llm_result.thought, source=self.agent_id))
            logger.debug(f"Agent {self.agent_id}: Received response from model '{self._model}'. Finish reason: {llm_result.finish_reason}")
        except Exception as llm_error:
            # Catch errors during the actual LLM call
            msg = f"Agent {self.agent_id}: Error during LLM call to '{self._model}': {llm_error}"
            logger.error(msg)
            error_output = ErrorEvent(source=self.agent_id, content=msg)
            # Wrap in AgentOutput
            return AgentOutput(agent_id=self.agent_id,
                metadata={"error": True},
                outputs=error_output,
            )
            # Depending on severity, might want to raise
            # raise ProcessingError(f"LLM call failed for agent {self.id}") from llm_error

        # 3. Create the standardized AgentTrace
        trace = self.make_trace(
            chat_result=llm_result,
            inputs=message,  # Pass AgentInput
            messages=llm_messages,  # Pass messages sent
            schema=self._output_model,  # Pass schema used for validation attempt
        )
        # Add agent role/name for context in logs/outputs
        trace.metadata.update({"role": self.role, "name": self.name})

        # 4. Wrap the trace in an AgentOutput
        response = AgentOutput(agent_id=self.agent_id,
            metadata=trace.metadata,
            outputs=trace.outputs,
        )

        logger.debug(f"Agent {self.agent_id} finished _process.")
        return response

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Resets agent state (e.g., task index, last input)."""
        # Inherited from Agent, can be extended by subclasses if they have more state.
        await super().on_reset(cancellation_token)
        # TODO: Review if _current_task_index and _last_input are used consistently or needed here.
        # self._current_task_index = 0
        # self._last_input = None
        logger.debug(f"LLMAgent {self.agent_id} state reset.")
