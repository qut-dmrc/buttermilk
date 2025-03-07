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
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from buttermilk._core.runner_types import RecordInfo
from buttermilk.bm import BM, logger
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    KeyValueCollector,
    _parse_prompty,
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
    agent_id: str
    role: str
    body: str | list | dict


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


class MessagesCollector(KeyValueCollector):
    """Specifically typed to collect pairs of (User, Assistant) messages"""

    _data: dict[str, list[UserMessage | AssistantMessage]] = PrivateAttr(
        default_factory=dict,
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

        self.step = step_name
        self.params = parameters
        self.template = template
        self._json_parser = ChatParser()
        self._model_client = bm.llms.get_autogen_chat_client(model)
        self._inputs = inputs

        # Generally, in a group chat, you want most of the information to be publicly
        # visible. These collectors provide storage for agents, usually by reading from the
        # the group chat log of messages. Useful for small objects predominantly.
        self._placeholders: KeyValueCollector = MessagesCollector()
        self._collected_vars: KeyValueCollector = KeyValueCollector()
        self._context = UnboundedChatCompletionContext()

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
        rendered_template = load_template(
            template=self.template,
            parameters=self.params,
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
                var_name = re.sub(r"[\W]+", "", message["content"])
                try:
                    messages.extend(combined_placeholders[var_name])
                except KeyError as e:
                    raise ValueError(
                        f"Missing placeholder {var_name} in template placeholder.",
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

        return messages

    async def query(
        self,
        additional_inputs: dict[str, str] = {},
    ) -> Answer:
        untrusted_vars = dict(**additional_inputs)
        untrusted_vars.update(self._collected_vars.get_dict())

        messages = await self.fill_template(
            untrusted_inputs=untrusted_vars,
            placeholder_messages=self._placeholders.get_dict(),
            context=await self._context.get_messages(),
        )

        response = await self._model_client.create(messages=messages)

        output = self._json_parser.parse(response.content)

        answer = Answer(
            agent_id=self.id.type,
            role="assistant",
            body=output,
            content=response.content,
            step=str(self.step),
            metadata=response.model_dump(exclude=["content"]),
        )
        return answer

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

        answer = await self.query(additional_inputs=dict(message.model_extra))

        await self.publish(answer)

        return answer

    @message_handler
    async def handle_answer(
        self,
        message: Answer,
        ctx: MessageContext,
    ) -> None:
        if message.step in self._inputs:
            draft = dict(agent_id=message.agent_id, body=message.body)
            self._collected_vars.add(message.step, draft)
            msg = UserMessage(
                content=str(message.body),
                source=message.agent_id,
            )
            await self._context.add_message(msg)

            logger.debug(
                f"LLMAgent {self.id} received and stored {message.type} from step {message.step}.",
            )

    @message_handler
    async def handle_inputrecords(
        self,
        message: InputRecord,
        ctx: MessageContext,
    ) -> None:
        if message.step in self._inputs:
            # Special handling for input records
            # text only for now.
            # TODO @nicsuzor: make multimodal again.

            msg = UserMessage(
                content=message.payload.fulltext,
                source=ctx.sender.type if ctx.sender else self.id.type,
            )
            self._placeholders.add("record", msg)

            logger.debug(
                f"LLMAgent {self.id} received and stored {message.type} from step {message.step}.",
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        self._placeholders = []
        await self._context.clear()
