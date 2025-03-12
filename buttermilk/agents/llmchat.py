from typing import Any

import regex as re
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat

from buttermilk._core.agent import AgentConfig
from buttermilk.bm import BM
from buttermilk.runner.chat import (
    Answer,
    BaseGroupChatAgent,
    NullAnswer,
    RequestToSpeak,
)
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
)


class LLMAgent(BaseGroupChatAgent):
    def __init__(
        self,
        *,
        config: AgentConfig,
        group_chat_topic_type: str = "default",
        fail_on_unfilled_parameters: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            group_chat_topic_type=group_chat_topic_type,
        )
        bm = BM()
        self._json_parser = ChatParser()
        self._model_client = bm.llms.get_autogen_chat_client(self.parameters["model"])
        self._fail_on_unfilled_parameters = fail_on_unfilled_parameters

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
                except KeyError as e:
                    raise ValueError(
                        f"Missing {var_name} in template or placeholder vars.",
                    ) from e
                continue

            # Check if there's content in the message after filling the template
            if re.sub(r"\s+", "", str(message["content"])):
                if message["role"] in ("system", "developer"):
                    messages.append(SystemMessage(content=message["content"]))
                elif message["role"] in ("assistant"):
                    messages.append(
                        AssistantMessage(
                            content=message["content"],
                            source=self.id.type,
                        ),
                    )
                else:
                    messages.append(
                        UserMessage(content=message["content"], source=self.id.type),
                    )

        if unfilled_vars and self._fail_on_unfilled_parameters:
            raise ValueError(
                f"Template has unfilled parameters: {', '.join(unfilled_vars)}",
            )

        return messages

    async def query(
        self,
        request: RequestToSpeak,
    ) -> Answer | NullAnswer:
        untrusted_vars = dict(**request.inputs)

        # History variable is a list of strings, but context variable is a
        # list of messages
        untrusted_vars["history"] = [
            f"{x.source}: {x.content}" for x in request.context
        ]
        placeholders = {"context": request.context}
        placeholders.update(request.placeholders)
        messages = await self.fill_template(
            untrusted_inputs=untrusted_vars,
            placeholder_messages=placeholders,
        )

        response = await self._model_client.create(messages=messages)

        output = self._json_parser.parse(response.content)

        answer = Answer(
            agent_id=self.id.type,
            role="assistant",
            content=response.content,
            step=str(self.step),
            metadata=response.model_dump(exclude=["content"]),
            config=self.config,
            inputs=untrusted_vars,
            outputs=output,
            context=request.context,
        )
        return answer
