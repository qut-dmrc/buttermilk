import asyncio
from curses import meta
import json
import pprint
from types import NoneType
from typing import Any, AsyncGenerator, Callable, Optional, Self

from autogen_core.models._types import UserMessage
from psutil import Process
import pydantic
import regex as re
from autogen_core import CancellationToken, FunctionCall, MessageContext
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, PrivateAttr
import weave

from buttermilk._core.agent import Agent, AgentInput, AgentOutput, ConductorResponse
from buttermilk._core.contract import (
    AllMessages,
    ConductorRequest,
    FlowMessage,
    GroupchatMessageTypes,
    LLMMessage,
    ToolOutput,
    UserInstructions,
    TaskProcessingComplete,
    ProceedToNextTaskSignal,
    OOBMessages,
)
from buttermilk._core.exceptions import ProcessingError

# Import only necessary classes from _core.llms
from buttermilk._core.llms import AutoGenWrapper, CreateResult, ModelOutput
from buttermilk._core.types import Record

# Restore original bm import
from buttermilk.bm import bm, logger
from buttermilk.utils._tools import create_tool_functions
from buttermilk.utils.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
    make_messages,
)


class LLMAgent(Agent):
    """
    An LLM agent that uses a Buttermilk template/model configuration (via LLMAgent).
    It expects input conforming to AgentInput (handled by LLMAgent/Agent base)
    and aims to output results conforming to AgentReasons.
    """

    fail_on_unfilled_parameters: bool = Field(default=True)
    _tools_list: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(
        default_factory=list,
    )
    _model: str = ""
    _name_components: list[str] = ["base_name", "_id", "_model"]
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: AutoGenWrapper = PrivateAttr()
    _output_model: Optional[type[BaseModel]] = None
    _pause: bool = PrivateAttr(default=False)

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        self._model = self.parameters.get("model")
        if self._model:
            self._model_client = bm.llms.get_autogen_chat_client(
                self._model,
            )
        else:
            raise ValueError("Must provide a model in the parameters.")

        return self

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        self._tools_list = create_tool_functions(self.tools)

        return self

    async def _fill_template(
        self,
        task_params: dict[str, Any],  # Accepts task-specific parameters
        inputs: dict[str, Any],
        context: list[LLMMessage] = [],
        records: list[Record] = [],
    ) -> list[Any]:
        """Fill the template with the given inputs and return a list of messages."""
        template = self.parameters.get("template", task_params.get("template", inputs.get("template")))
        if not template:
            raise ProcessingError(f"No template provided for agent {self.id}")

        # Render the template using Jinja2
        rendered_template, unfilled_vars = load_template(
            template=template,
            parameters=task_params,  # Use combined parameters for template source
            untrusted_inputs=inputs,  # Use derived jinja_vars for filling template
        )

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys. First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to divide into messages and set roles
        messages = make_messages(local_template=prompty, context=context, records=records)

        if unfilled_vars := (set(unfilled_vars) - set(["records", "context"])):
            err = f"Template for agent {self.role} has unfilled parameters: {', '.join(unfilled_vars)}"
            if self.fail_on_unfilled_parameters:
                raise ProcessingError(err)
        return messages

    def make_output(
        self,
        chat_result: CreateResult,
        inputs: AgentInput | None = None,
        messages: list[LLMMessage] = [],
        parameters: dict = {},
        prompt: str = "",
        schema: Optional[type[BaseModel]] = None,
    ) -> AgentOutput:
        """Helper to create AgentOutput from CreateResult."""
        output = AgentOutput(
            agent_id=self.id,
            inputs=inputs,
            messages=messages,
            params=parameters,
            prompt=prompt,
            metadata=chat_result.model_dump(exclude={"content", "object"}),
        )

        # Handle schema validation first if applicable
        if schema and isinstance(chat_result, ModelOutput) and isinstance(chat_result.object, schema):
            output.outputs = chat_result.object
            return output

        elif schema and isinstance(chat_result.content, str):
            try:
                # model_validate_json returns an instance of the schema model
                output.outputs = schema.model_validate_json(chat_result.content)
                return output
            except Exception as e:
                error = f"Error parsing response from LLM: {e} into {schema.__name__}"
                logger.warning(error, exc_info=False)
                output.error.append(error)
                # Fall through to default JSON parsing or raw content handling

        # Default handling: attempt JSON parsing or store raw content
        if isinstance(chat_result.content, str):
            try:
                # Store parsed dict in outputs
                output.outputs = self._json_parser.parse(chat_result.content)
            except Exception as parse_error:
                # If JSON parsing fails, store raw string content
                error = f"Failed to parse LLM response as JSON: {parse_error}. Storing raw content."
                logger.warning(error, exc_info=False)
                output.error.append(error)
                output.outputs = chat_result.content

        elif chat_result.content is not None:  # Handle non-string, non-None content if necessary
            logger.warning(f"LLM response content is not a string: {type(chat_result.content)}. Storing as is.")
            output.outputs = chat_result.content

        return output

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput:
        """Runs a single task or series of tasks."""
        try:
            messages = await self._fill_template(
                task_params=message.parameters, inputs=message.inputs, context=message.context, records=message.records
            )
        except ProcessingError as e:
            # We log here because weave swallows error results and we might lose it.
            logger.error(f"Unable to fill template for {self.id}: {e}")
            raise

        llm_result: CreateResult = await self._model_client.call_chat(
            messages=messages,
            tools_list=self._tools_list,
            cancellation_token=cancellation_token,
            schema=self._output_model,
        )

        output = self.make_output(llm_result, inputs=message.inputs, messages=messages, parameters=message.parameters, schema=self._output_model)
        output.metadata.update({"role": self.role, "name": self.name})
        output.prompt = message.prompt

        return output

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        await super().on_reset(cancellation_token)
        self._current_task_index = 0
        self._last_input = None
        logger.debug(f"Agent {self.role} state reset.")
