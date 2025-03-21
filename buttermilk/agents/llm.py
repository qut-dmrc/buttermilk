from typing import Any, Self

import pydantic
import regex as re
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat
from pydantic import Field, PrivateAttr

from buttermilk._core.agent import Agent, AgentInput, AgentOutput
from buttermilk._core.contract import AgentMessages
from buttermilk.bm import bm, logger
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
)


class LLMAgent(Agent):
    fail_on_unfilled_parameters: bool = Field(default=False)

    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: ChatCompletionClient = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        if self.parameters.get("model"):
            self._model_client = bm.llms.get_autogen_chat_client(self.parameters["model"])

        return self

    async def receive_output(
        self,
        message: AgentMessages,
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
            untrusted_inputs = inputs.inputs
            untrusted_inputs["prompt"] = inputs.prompt

        # Render the template using Jinja2
        rendered_template, unfilled_vars = load_template(
            parameters=self.parameters, untrusted_inputs=untrusted_inputs,
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
                        messages.append(UserMessage(content=rec.fulltext, source="record"))
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
                        err = (f"Missing {var_name} in template or placeholder vars.",)
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
                            source=self.agent_id,
                        ),
                    )
                else:
                    messages.append(
                        UserMessage(content=content_without_vars, source=self.agent_id),
                    )

        if unfilled_vars:
            err = f"Template has unfilled parameters: {', '.join(unfilled_vars)}"
            if self.fail_on_unfilled_parameters:
                raise ValueError(err)
            logger.warning(err)

        return messages

    async def process(self, input_data: AgentInput, **kwargs) -> AgentOutput:

        messages = await self.fill_template(
            inputs=input_data,
        )

        response = await self._model_client.create(messages=messages)

        outputs = self._json_parser.parse(response.content)
        metadata = {
            k: v
            for k, v in response.model_dump(
                exclude_unset=True,
                exclude_none=True,
            ).items()
            if v and k != "content"
        }
        output = AgentOutput(agent_id=self.agent_id, response=outputs, content=response.content, metadata=metadata)
        return output
