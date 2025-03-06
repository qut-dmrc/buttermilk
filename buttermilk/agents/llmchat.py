import asyncio
from typing import Any

import regex as re
from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.model_context import (
    UnboundedChatCompletionContext,
)
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, model_validator

from buttermilk.bm import BM, logger
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import _parse_prompty, load_template_vars


class GroupChatMessage(BaseModel):
    type: str = "GroupChatMessage"
    """A message sent to the group chat"""

    content: str | LLMMessage
    """The content of the message."""

    source: str
    """The name of the agent that sent this message."""

    step: str
    """The stage of the process that this message was sent from"""

    data: dict[str, Any] = {}
    """Additional data related to the message."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about the message."""

    @model_validator(mode="before")
    @classmethod
    def coerce_content_to_string(cls, values):
        if "content" in values:
            content = values["content"]
            if isinstance(content, str):
                pass  # Already a string
            elif hasattr(content, "content"):  # Handle LLMMessage case
                values["content"] = str(content.content)
            else:
                values["content"] = str(content)
        return values


class Request(GroupChatMessage):
    type: str = "Request"


class Answer(GroupChatMessage):
    type: str = "Answer"


class RequestToSpeak(BaseModel):
    model_config = {"extra": "allow"}


class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str = "default",
    ):
        super().__init__(
            description=description,
        )
        self._group_chat_topic_type = group_chat_topic_type
        self._json_parser = ChatParser()

    async def publish(self, message: Any) -> None:
        await self.publish_message(
            message,
            # DefaultTopicId(type=self._group_chat_topic_type, source=self.step),
            DefaultTopicId(type=self._group_chat_topic_type),
        )


class LLMAgent(BaseGroupChatAgent):
    def __init__(
        self,
        template: str,
        model: str,
        name: str,
        *,
        description: str,
        step_name: str,
        inputs: list[str] = [],
        group_chat_topic_type: str = "default",
        **parameters,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
        )
        bm = BM()

        self._context = UnboundedChatCompletionContext()
        self.step = step_name
        self.params = parameters
        self.template = template
        self._json_parser = ChatParser()
        self._model_client = bm.llms.get_autogen_chat_client(model)
        self._name = name
        self._inputs = inputs
        self._history: list[str] = []

    async def load_template(self, inputs: dict[str, Any] = {}) -> list[Any]:
        # Construct list of messages from the templates
        rendered_template = load_template_vars(
            template=self.template,
            parameters=self.params,
            untrusted_inputs=inputs,
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
            # Check if there's content in the message after filling the template
            if re.sub(r"\s+", "", str(message["content"])):
                if message["role"] in ("system", "developer"):
                    messages.append(SystemMessage(content=message["content"]))
                elif message["role"] in ("assistant"):
                    messages.append(
                        AssistantMessage(content=message["content"], source=self._name),
                    )
                else:
                    messages.append(
                        UserMessage(content=message["content"], source=self._name),
                    )

        return messages

    async def query(
        self,
        inputs: dict[str, Any] = {},
    ) -> CreateResult:
        messages = await self.load_template(inputs=inputs)
        await asyncio.sleep(1)
        messages.extend(await self._context.get_messages())
        messages.extend(self._history)

        response = await self._model_client.create(messages=messages)

        return response

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> GroupChatMessage:
        log_message = f"{self._name} from {self.step} got request to speak."

        if message.model_extra:
            log_message += f" Args: {', '.join(message.model_extra.keys())}."

        logger.debug(log_message)

        response = await self.query(inputs=message.model_extra)
        output = self._json_parser.parse(response.content)

        answer = GroupChatMessage(
            content=output,
            source=self._name,
            step=str(self.step),
            metadata=response.model_dump(exclude=["content"]),
        )
        await self.publish(answer)

        return answer

    @message_handler
    async def handle_groupchatmessage(
        self,
        message: Request | GroupChatMessage | Answer,
        ctx: MessageContext,
    ) -> None:
        # Process the message using the LLM client
        if message.step in self._inputs:
            self._history.append(message.content)
            await self._context.add_message(message.content)

            logger.debug(
                f"LLMAgent {self._name} received step {message.step} message from {message.source}",
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        self._history = []
        await self._context.clear()
