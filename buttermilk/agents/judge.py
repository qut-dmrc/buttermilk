import asyncio
from curses import meta
import json
import pprint
from types import NoneType
from typing import Any, AsyncGenerator, Callable, Optional, Self

from autogen_core.models._types import UserMessage
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
from buttermilk._core.llms import AutoGenWrapper, CreateResult
from buttermilk._core.types import Record
from buttermilk.bm import bm, logger
from buttermilk.utils._tools import create_tool_functions
from buttermilk.utils.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
    make_messages,
)

class Reasons(BaseModel):
    conclusion: str
    reasoning: str


class LLMAgent(Agent):
    fail_on_unfilled_parameters: bool = Field(default=True)
    _tools_list: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(
        default_factory=list,
    )
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: AutoGenWrapper = PrivateAttr()
    _output_model: Optional[type[BaseModel]] = None
    _pause: bool = PrivateAttr(default=False)

    # @pydantic.model_validator(mode="after")
    # def custom_agent_id(self) -> Self:
    #     # Set a custom name based on our major variants
    #     components = self.role.split("-")

    #     components.extend(
    #         [v for k, v in self.variants.items() if k not in ["formatting", "description", "template"] and v and not re.search(r"\s", v)]
    #     )
    #     components = [c[:12] for c in components if c]
    #     self.role = "_".join(components)[:63]

    #     return self

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        if self.parameters.get("model"):
            self._model_client = bm.llms.get_autogen_chat_client(
                self.parameters["model"],
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
            raise ProcessingError("No template provided for agent {self.id}")

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

    def make_output(self, chat_result: CreateResult, inputs: AgentInput, schema: Optional[type[BaseModel]] = None) -> AgentOutput:
        output = AgentOutput(role=self.role)
        output.inputs = inputs.model_copy(deep=True)
        output.metadata = chat_result.model_dump(exclude="content")

        model_metadata = self.parameters
        model_metadata.update({"role": self.role, "agent_id": self.id, "name": self.name, "prompt": inputs.prompt})
        model_metadata.update(inputs.parameters)
        output.params = model_metadata

        if schema:
            try:
                output.outputs = schema.model_validate_json(chat_result.content)
                return output
            except Exception as e:
                error = f"Error parsing response from LLM: {e} into {type(schema)}"
                logger.warning(error, exc_info=False)

        if isinstance(chat_result.content, str):
            try:
                output.outputs = self._json_parser.parse(chat_result.content)
            except Exception as parse_error:
                error = f"Failed to parse LLM response as JSON: {parse_error}"
                logger.warning(error, exc_info=False)
                output.outputs = dict(response=chat_result.content)

        output.content = pprint.pformat(output.outputs)
        return output

    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken = None, **kwargs) -> AgentOutput | ToolOutput | None:
        """Runs a single task or series of tasks."""

        messages = await self._fill_template(task_params=inputs.parameters, inputs=inputs.inputs, context=inputs.context, records=inputs.records)
        result = await self._model_client.call_chat(
            messages=messages,
            tools_list=self._tools_list,
            cancellation_token=cancellation_token,
            reflect_on_tool_use=True,
            schema=self._output_model,
        )
        result = self.make_output(result, inputs=inputs, schema=self._output_model)

        return result

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        await super().on_reset(cancellation_token)
        self._current_task_index = 0
        self._last_input = None
        logger.debug(f"Agent {self.role} state reset.")
