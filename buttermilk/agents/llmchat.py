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
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, model_validator

from buttermilk._core.runner_types import RecordInfo
from buttermilk.bm import BM, logger
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    KeyValueCollector,
    _parse_prompty,
    finalise_template,
    load_template,
)


class GroupChatMessage(BaseModel):
    type: str = "GroupChatMessage"
    """A message sent to the group chat"""

    content: str
    """The content of the message."""

    step: str
    """The stage of the process that this message was sent from"""

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


class InputRecord(GroupChatMessage):
    type: str = "InputRecord"
    payload: RecordInfo = Field(
        ...,
        description="A single instance of an input example for workers to use.",
    )


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
        self._json_parser = ChatParser()
        self._model_client = bm.llms.get_autogen_chat_client(model)
        self._inputs = inputs

        # Use this sparingly. Generally, in a group chat, you want most of the information to be publicly
        # visible. This provides a way to send additional chunks of data to particular agents that will not
        # show up in the group chat log of messages. Useful for medium to large objects particularly.
        # So far we mainly use this to transfer the canonical RecordInfo object as an input.
        self._data: KeyValueCollector = KeyValueCollector()

        self.partial_template = load_template(
            template=template,
            parameters=self.params,
        )

    async def fill_template(
        self,
        placeholder_messages: dict[
            str,
            list[SystemMessage | UserMessage | AssistantMessage],
        ] = {},
        untrusted_inputs: dict[str, Any] = {},
    ) -> list[Any]:

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys. First we strip the header information from the markdown
        prompty = _parse_prompty(self.partial_template)

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
            # For placeholder messages, we are subbing in one or more
            # entire Message objects
            if message["role"] == "placeholder":
                # Remove everything except word chars to get the variable name
                var_name = re.sub(r"[\W]+", "", message["content"])
                try:
                    messages.extend(placeholder_messages[var_name])
                except KeyError as e:
                    raise ValueError(
                        f"Missing placeholder {var_name} in template placeholder.",
                    ) from e
                continue

            # For all message types except Placeholder, we fill the template
            # using jinja2 substitution in a sandboxed environment (text only).
            message["content"] = finalise_template(
                intermediate_template=str(message["content"]),
                untrusted_inputs=untrusted_inputs,
            )

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

        return messages

    async def query(
        self,
        inputs: dict[str, Any] = {},
    ) -> CreateResult:
        messages = await self.fill_template(
            untrusted_inputs=inputs,
            placeholder_messages=self._data.get_dict(),
        )
        # messages.extend(await self._context.get_messages())

        response = await self._model_client.create(messages=messages)

        return response

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> Answer:
        log_message = f"{self.id} from {self.step} got request to speak."

        if message.model_extra:
            log_message += f" Args: {', '.join(message.model_extra.keys())}."

        logger.debug(log_message)

        inputs = {}
        inputs["context"] = await self._context.get_messages()
        inputs.update(message.model_extra)

        response = await self.query(inputs=inputs)
        output = self._json_parser.parse(response.content)

        answer = Answer(
            content=output,
            step=str(self.step),
            metadata=response.model_dump(exclude=["content"]),
        )
        await self.publish(answer)

        return answer

    @message_handler
    async def handle_groupchatmessage(
        self,
        message: InputRecord | GroupChatMessage | Answer,
        ctx: MessageContext,
    ) -> None:
        # Process the message using the LLM client
        if message.step in self._inputs:
            if isinstance(message, InputRecord):
                # Special handling for input records
                # text only for now.
                # TODO @nicsuzor: make multimodal again.

                msg = UserMessage(
                    content=message.payload.fulltext,
                    source=ctx.sender.type if ctx.sender else self.id.type,
                )
                self._data.add("record", msg)
            else:
                self._data.add("context", message.content)
                await self._context.add_message(message.content)

            logger.debug(
                f"LLMAgent {self.id} received and stored {message.type} from step {message.step}.",
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        self._data = []
        await self._context.clear()
