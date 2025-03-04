from collections.abc import Coroutine
from typing import Any

import shortuuid
from autogen_core import (
    CancellationToken,
    MessageContext,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat

from buttermilk.bm import BM, logger
from buttermilk.runner.moa import (
    BaseGroupChatAgent,
    GroupChatMessage,
    Request,
    RequestToSpeak,
)
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import _parse_prompty, load_template_vars


class LLMAgent(BaseGroupChatAgent):
    def __init__(
        self,
        *,
        llm: str,
        template: str,
        name: str | None = None,
        **data,
    ) -> None:
        super().__init__(**data)

        self._unfilled_inputs: list[str] = []
        self._context: list[str] = []
        # _model_context = UnboundedChatCompletionContext()
        self._template = template
        self._json_parser = ChatParser()

        self._model_client = BM().llms.get_autogen_client(llm)
        self._name = "-".join([
            x for x in [name, template, llm, shortuuid.uuid()[:6]] if x
        ])

    def load_template(self, params: dict[str, str]) -> list[Any]:
        # Construct list of messages from the templates
        rendered_template, self._unfilled_inputs = load_template_vars(
            template=self._template,
            **params,
        )

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys. First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to set roles within the template
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
            if message["role"] == "placeholder":
                if message["content"].strip("{}") in self._unfilled_inputs:
                    continue
                messages.append(
                    UserMessage(content=message["content"], source=self._name),
                )
            elif message["role"] in ("system", "developer"):
                messages.append(SystemMessage(content=message["content"]))
            elif message["role"] in ("assistant"):
                messages.append(
                    AssistantMessage(content=message["content"], source=self._name),
                )
            else:
                messages.append(
                    UserMessage(content=message["content"], source=self._name),
                )

        messages.extend(self._context)

        return messages

    async def query(
        self,
        params: dict[str, Any] = {},
    ) -> Coroutine[Any, Any, CreateResult]:
        messages = self.load_template(params=params)

        response = await self._model_client.create(messages=messages)

        return response

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> GroupChatMessage:
        info_message = f"{self._name} from {self._step} got request to speak."

        if message.model_extra:
            info_message += f" Args: {', '.join(message.model_extra.keys())}."

        logger.debug(info_message)

        response = await self.query(params=message.model_extra)
        output = self._json_parser.parse(response.content)

        answer = GroupChatMessage(
            content=output,
            source=self._name,
            step=str(self._step),
            metadata=response.model_dump(exclude=["content"]),
        )
        await self.publish(answer)
        return answer

    @message_handler
    async def handle_groupchatmessage(
        self,
        message: Request,
        ctx: MessageContext,
    ) -> None:
        # Process the message using the LLM client
        if message.step in self.inputs:
            logger.debug(
                f"LLMAgent {self._name} received step {message.step} message from {message.source}",
            )
            # self._chat_history.append(message)
            # await self._context.add_message(message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        await self._context.clear()
