from collections.abc import Mapping
from typing import Any

import shortuuid
from autogen_core import (
    CancellationToken,
    MessageContext,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
)
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, PrivateAttr

from buttermilk.bm import logger
from buttermilk.runner.moa import (
    BaseGroupChatAgent,
    GroupChatMessage,
    Request,
    RequestToSpeak,
)
from buttermilk.utils.templating import _parse_prompty, load_template_vars


class LLMAgent(BaseGroupChatAgent, BaseModel):
    inputs: list[str] = []

    _llm_client: ChatCompletionClient = PrivateAttr()
    _template: str = PrivateAttr()
    _unfilled_inputs: list[str] = PrivateAttr()
    _context: list[str] = PrivateAttr(default=[])

    # _model_context = UnboundedChatCompletionContext()

    def __init__(
        self,
        *,
        llm_client: ChatCompletionClient,
        template: str,
        **data,
    ) -> None:
        BaseGroupChatAgent.__init__(**data)
        self._template = template
        self._llm_client = llm_client
        self._name = "-".join([
            x
            for x in [data.get("name"), template, llm_client, shortuuid.uuid()[:6]]
            if x
        ])

    async def save_state(self) -> dict[str, Any]:
        return self.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)

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
        messages = parse_chat(
            prompty,
            valid_roles=["system", "user", "developer", "human", "placeholder"],
        )

        messages.extend(self._context)

        return messages

    async def query(
        self,
        params: dict[str, Any] = {},
    ) -> str:
        messages = self.load_template(params=params)
        response = await self._llm_client.create(messages=messages)
        yield response.content

        return response

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> GroupChatMessage:
        info_message = (
            f"UserAgent {self.name} from step {self._step} received request to speak.",
        )

        if message.model_extra:
            info_message += "Args: {", ".join(message.model_extra.keys())}."

        logger.debug(info_message)

        response = await self.query(params=message.model_extra)

        answer = GroupChatMessage(
            content=response,
            source=self.name,
            step=self._step,
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
                f"LLMAgent {self.name} received step {message.step} message from {message.source}",
            )
            self._chat_history.append(message)
            # await self._context.add_message(message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        await self._context.clear()
