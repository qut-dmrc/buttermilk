import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Self

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
from pydantic import Field, PrivateAttr

from buttermilk._core.agent import Agent, AgentInput, AgentOutput
from buttermilk._core.contract import (
    AllMessages,
    ConductorRequest,
    FlowMessage,
    GroupchatMessageTypes,
    ToolOutput,
    UserInstructions,
    TaskProcessingComplete,
    ProceedToNextTaskSignal,
    OOBMessages,
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.runner_types import Record
from buttermilk.bm import bm, logger
from buttermilk.utils._tools import create_tool_functions
from buttermilk.utils.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
    make_messages,
)


class LLMAgent(Agent):
    fail_on_unfilled_parameters: bool = Field(default=True)
    _tools_list: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(
        default_factory=list,
    )
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: ChatCompletionClient = PrivateAttr()

    _pause: bool = PrivateAttr(default=False)

    @pydantic.model_validator(mode="after")
    def custom_agent_id(self) -> Self:
        # Set a custom name based on our major variants
        components = self.id.split("-")

        components.extend([
            v
            for k, v in self.variants.items()
            if k not in ["formatting", "description", "template"]
            and v and not re.search(r"\s", v)
        ])
        components = [c[:12] for c in components if c]
        self.id = "_".join(components)[:63]

        return self

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
        placeholders: dict[str, Any] = {},
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
        messages = make_messages(local_template=prompty, placeholders=placeholders)

        if (unfilled_vars := (set(unfilled_vars) - set(placeholders.keys()))):
            err = f"Template for agent {self.id} has unfilled parameters: {', '.join(unfilled_vars)}"
            if self.fail_on_unfilled_parameters:
                raise ProcessingError(err)
        return messages

    async def _execute_tools(
        self,
        calls: list[FunctionCall],
        cancellation_token: CancellationToken | None,
    ) -> list[ToolOutput]:
        """Execute the tools and return the results."""
        assert isinstance(calls, list) and all(
            isinstance(call, FunctionCall) for call in calls
        )

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._call_tool(call, cancellation_token) for call in calls],
        )
        results = [record for result in results for record in result if record is not None]

        return results

    async def _call_tool(
        self,
        call: FunctionCall,
        cancellation_token: CancellationToken | None,
    ) -> list[ToolOutput]:
        # Find the tool by name.
        tool = next((tool for tool in self._tools_list if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        arguments = json.loads(call.arguments)
        results = await tool.run_json(arguments, cancellation_token)

        if not isinstance(results, list):
            results = [results]
        outputs = []
        for result in results:
            result.call_id = call.id
            result.name = tool.name
            outputs.append(result)
        return outputs

    async def _create_agent_output(
        self,
        raw_content: str | list[Any],
        inputs: AgentInput,
        llm_metadata: dict = {},
        error_msg: str | None = None,
    ) -> AgentOutput:
        """Helper method to create AgentOutput instances."""
        outputs = {}
        content = ""
        if isinstance(raw_content, str):
            try:
                outputs = self._json_parser.parse(raw_content)
                content = json.dumps(outputs, indent=2, sort_keys=True)
            except Exception as parse_error:
                logger.warning(f"Failed to parse LLM response as JSON: {parse_error}")
                content = raw_content
                if not error_msg:
                    error_msg = f"JSON parsing error: {parse_error}"
        else:
            outputs = raw_content
            content = str(raw_content)
        dump = {k: v for k, v in inputs.__dict__.items() if k in inputs.model_fields_set}
        response = AgentOutput(**dump)
        response.content = content
        response.outputs = outputs
        response.error = error_msg
        response.metadata = llm_metadata
        return response

    async def _listen(
        self, message: GroupchatMessageTypes, cancellation_token: CancellationToken = None, publish_callback: Callable = None, **kwargs
    ) -> AsyncGenerator[GroupchatMessageTypes | None, None]:
        """Save incoming messages for later use."""
        if message.content:
            # Map Buttermilk message types to LLM input types
            if isinstance(message, AgentOutput):
                await self._model_context.add_message(AssistantMessage(content=str(message.content), source=message.source))
                if message.records:
                    self._records.extend(message.records)
            elif isinstance(message, (ToolOutput, UserInstructions)):
                await self._model_context.add_message(UserMessage(content=str(message.content), source=message.source))
            else:
                # don't log other types of messages
                pass
        yield None

    async def _process(
        self, inputs: AgentInput, cancellation_token: CancellationToken = None, publish_callback: Callable = None, **kwargs
    ) -> AsyncGenerator[AgentOutput | ToolOutput | None, None]:
        """Runs a single task or series of tasks."""
        placeholders = {
            "records": [rec.as_message() for rec in inputs.records if rec],
            "context": inputs.context,
        }
        messages = await self._fill_template(task_params=inputs.params, inputs=inputs.inputs, placeholders=placeholders)

        create_result = await self._model_client.create(messages=messages, tools=self._tools_list, cancellation_token=cancellation_token)
        llm_metadata = create_result.model_dump(exclude_unset=True, exclude_none=True)

        if isinstance(create_result.content, str):
            if create_result.content.strip() != "":
                yield await self._create_agent_output(
                    raw_content=create_result.content,
                    inputs=inputs,
                    llm_metadata=llm_metadata,
                )
        elif isinstance(create_result.content, list) and all(isinstance(item, FunctionCall) for item in create_result.content):
            tool_outputs = await self._execute_tools(
                calls=create_result.content,
                cancellation_token=cancellation_token,
            )
            reflection_tasks = []
            for tool_result in tool_outputs:
                if tool_result.is_error:
                    error_msg = f"Tool call '{tool_result.source}' failed: {tool_result.content}"
                    logger.warning(error_msg)
                    continue

                yield tool_result

                await asyncio.sleep(0.1)

                try:
                    reflection_messages = messages.copy()
                    reflection_messages.extend(tool_result.messages)
                    task = self._model_client.create(
                        messages=reflection_messages,
                        cancellation_token=cancellation_token,
                    )
                    reflection_tasks.append(task)
                except Exception as e:
                    error_msg = f"Error preparing reflection for tool '{tool_result.source}': {e}"
                    logger.warning(error_msg, exc_info=True)
                    raise ProcessingError(error_msg)

            for task in asyncio.as_completed(reflection_tasks):
                try:
                    reflection_result = await task
                    reflection_metadata = reflection_result.model_dump(exclude_unset=True, exclude_none=True)
                    yield await self._create_agent_output(
                        raw_content=reflection_result.content,
                        inputs=inputs,
                        llm_metadata=reflection_metadata,
                    )
                except Exception as e:
                    error_msg = f"Error during reflection LLM call: {e}"
                    logger.warning(error_msg, exc_info=False)
                    raise ProcessingError(error_msg)
        else:
            error_msg = f"Unexpected content type from LLM (task): {type(create_result.content)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
        return

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        await super().on_reset(cancellation_token)
        self._current_task_index = 0
        self._last_input = None
        logger.debug(f"Agent {self.id} state reset.")
