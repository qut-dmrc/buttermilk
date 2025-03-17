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

    # @pydantic.model_validator(mode="after")
    # def init_model(self) -> Self:
    #     self._model_client = bm.llms.get_autogen_chat_client(self.model)

    #     return self

    async def fill_template(
        self,
        placeholder_messages: dict[
            str,
            list[SystemMessage | UserMessage | AssistantMessage],
        ] = {},
        untrusted_inputs: dict[str, Any] = {},
        context: list[SystemMessage | UserMessage | AssistantMessage] = [],
    ) -> list[Any]:
        """Fill the template with the given inputs and return a list of messages."""
        # Render the template using Jinja2
        rendered_template, unfilled_vars = load_template(
            parameters=self.parameters,
            untrusted_inputs=untrusted_inputs,
        )

        combined_placeholders = dict(context=context)
        combined_placeholders.update(placeholder_messages)

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
                try:
                    messages.extend(combined_placeholders[var_name])
                    # Remove the placeholder from the list of unfilled variables
                    if var_name in unfilled_vars:
                        unfilled_vars.remove(var_name)
                except KeyError:
                    err =  f"Missing {var_name} in template or placeholder vars.",
                    if self.fail_on_unfilled_parameters:
                        raise ValueError(err)
                    else:
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
                            source=self.id.type,
                        ),
                    )
                else:
                    messages.append(
                        UserMessage(content=content_without_vars, source=self.id.type),
                    )

        if unfilled_vars:
            err = f"Template has unfilled parameters: {', '.join(unfilled_vars)}"
            if self.fail_on_unfilled_parameters:
                raise ValueError(err)
            else:
                logger.warning(err)


        return messages

    async def process(self, input_data: AgentInput) -> AgentOutput:

        untrusted_vars= input_data.model_copy()

        messages = await self.fill_template(
            untrusted_inputs=untrusted_vars
        )

        response = await self._model_client.create(messages=messages)

        outputs = self._json_parser.parse(response.content)

        return outputs
