import asyncio
import json
from typing import Any, Self

import pydantic
import regex as re
from autogen_core import CancellationToken, FunctionCall
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
from buttermilk._core.contract import AgentMessages, ToolOutput, UserInput
from buttermilk._core.runner_types import Record
from buttermilk.bm import bm, logger
from buttermilk.runner.helpers import create_tool_functions
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
)


class LLMAgent(Agent):
    fail_on_unfilled_parameters: bool = Field(default=True)

    _tools: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(
        default_factory=list,
    )
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: ChatCompletionClient = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        if self.parameters.get("model"):
            self._model_client = bm.llms.get_autogen_chat_client(
                self.parameters["model"],
            )
        else:
            raise ValueError("Must provide a model in the parameters.")

        # Initialise tools
        self._tools = create_tool_functions(self.tools)
        return self

    async def receive_output(
        self,
        message: AgentMessages | UserInput,
        source: str,
        **kwargs,
    ) -> AgentMessages | None:
        """Log data or send output to the user interface"""
        # Not implemented on the agent right now; inputs come from conductor.

    async def fill_template(
        self,
        inputs: AgentInput | None = None,
    ) -> list[Any]:
        """Fill the template with the given inputs and return a list of messages."""
        untrusted_inputs = {}
        if inputs:
            untrusted_inputs.update(dict(inputs.inputs))

        # Render the template using Jinja2
        rendered_template, unfilled_vars = load_template(
            parameters=self.parameters,
            untrusted_inputs=untrusted_inputs,
        )

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys. First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to divide into messages and set roles
        messages = []
        for message in parse_chat(
            prompty,
            valid_roles=[
                "system",
                "user",
                "developer",
                "human",
                "placeholder",
                "assistant",
            ],
        ):
            # For placeholder messages, we are subbing in one or more
            # entire Message objects
            if message["role"] == "placeholder":
                # Remove everything except word chars to get the variable name
                var_name = re.sub(r"[^\w\d_]+", "", message["content"])
                if var_name.lower() == "records" and inputs:
                    for rec in inputs.records:
                        # TODO make this multimodal later
                        messages.append(
                            UserMessage(content=rec.fulltext, source="record"),
                        )
                    # Remove the placeholder from the list of unfilled variables
                    if var_name in unfilled_vars:
                        unfilled_vars.remove(var_name)
                else:
                    try:
                        messages.extend(getattr(inputs, var_name))
                        # Remove the placeholder from the list of unfilled variables
                        if var_name in unfilled_vars:
                            unfilled_vars.remove(var_name)
                    except KeyError:
                        err = (
                            f"Missing {var_name} in template or placeholder vars for agent {self.id}.",
                        )
                        if self.fail_on_unfilled_parameters:
                            raise ValueError(err)
                        logger.warning(err)

                continue

            # Remove unfilled variables now
            content_without_vars = re.sub(r"\{\{.*?\}\}", "", message["content"])

            # And check if there's content in the message still
            if re.sub(r"\s+", "", content_without_vars):
                if message["role"] in ("system", "developer"):
                    messages.append(SystemMessage(content=content_without_vars))
                elif message["role"] in ("assistant"):
                    messages.append(
                        AssistantMessage(
                            content=content_without_vars,
                            source=self.id,
                        ),
                    )
                else:
                    messages.append(
                        UserMessage(content=content_without_vars, source=self.id),
                    )

        if unfilled_vars:
            err = f"Template for agent {self.id} has unfilled parameters: {', '.join(unfilled_vars)}"
            if self.fail_on_unfilled_parameters:
                raise ValueError(err)
            logger.warning(err)

        return messages

    async def _execute_tools(
        self,
        calls: list[FunctionCall],
        cancellation_token: CancellationToken | None,
    ) -> dict[str, Any]:
        """Execute the tools and return the results."""
        assert isinstance(calls, list) and all(
            isinstance(call, FunctionCall) for call in calls
        )

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._call_tool(call, cancellation_token) for call in calls],
        )
        outputs = {"_records": []}
        for r in results:
            if r.is_error:
                logger.warning(f"Tool {r.name} failed with error: {r.content}")
                continue
            if r.payload and isinstance(r.payload, Record):
                outputs["_records"].append(r.payload)
            else:
                outputs[r.name] = r.content

        return outputs

    async def _call_tool(
        self,
        call: FunctionCall,
        cancellation_token: CancellationToken | None,
    ) -> ToolOutput:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return ToolOutput(
                call_id=call.id,
                content=tool.return_value_as_string(result),
                payload=result,
                is_error=False,
                name=tool.name,
            )
        except Exception as e:
            return ToolOutput(
                call_id=call.id,
                content=str(e),
                is_error=True,
                name=tool.name,
                payload=e,
            )

    async def _process(
        self,
        input_data: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AgentOutput:
        messages = await self.fill_template(
            inputs=input_data,
        )
        records = []
        try:
            create_result = await self._model_client.create(
                messages=messages,
                tools=self._tools,
                cancellation_token=cancellation_token,
            )
        except Exception as e:
            error_msg = f"Error creating chat completion: {e} {e.args=}"
            logger.warning(error_msg)
            return AgentOutput(
                agent_id=self.id,
                agent_name=self.name,
                content=error_msg,
                metadata=dict(self.parameters),
            )
        if isinstance(create_result.content, str):
            outputs = self._json_parser.parse(create_result.content)

            # create human readable content
            # Pretty-print with standard library
            content = json.dumps(outputs, indent=2, sort_keys=True)
        else:
            outputs = await self._execute_tools(
                calls=create_result.content,
                cancellation_token=cancellation_token,
            )
            records = outputs.pop("_records", [])
            content = str(outputs)

        metadata = {
            k: v
            for k, v in create_result.model_dump(
                exclude_unset=True,
                exclude_none=True,
            ).items()
            if v and k != "content"
        }
        metadata.update(self.parameters)
        output = AgentOutput(
            agent_id=self.id,
            agent_name=self.name,
            outputs=outputs,
            content=content,
            metadata=metadata,
            records=records,
        )
        return output
