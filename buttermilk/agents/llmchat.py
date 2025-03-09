from collections.abc import Mapping
from typing import Any

import regex as re
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from buttermilk._core.agent import AgentConfig
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

    inputs: dict = {}
    outputs: dict = {}
    context: list[SystemMessage | UserMessage | AssistantMessage] = []

    config: AgentConfig

    model_config = {"extra": "allow"}


class RequestToSpeak(BaseModel):
    inputs: Mapping[str, Any] = {}
    placeholders: Mapping[
        str,
        list[SystemMessage | UserMessage | AssistantMessage],
    ] = {}
    context: list[SystemMessage | UserMessage | AssistantMessage] = []


class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str = "default",
    ) -> None:
        """Initialize the agent with configuration and topic type.

        Args:
            config: Configuration settings for the agent
            group_chat_topic_type: The type of group chat topic to use (default: "default")

        """
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
        *,
        config: AgentConfig,
        group_chat_topic_type: str = "default",
        fail_on_unfilled_parameters: bool = True,
    ) -> None:
        super().__init__(
            description=config.description,
            group_chat_topic_type=group_chat_topic_type,
        )
        bm = BM()
        self.config = config
        self.step = config.name
        self.parameters = config.parameters
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
        inputs: dict[str, str] = {},
        placeholders: dict[
            str,
            list[SystemMessage | UserMessage | AssistantMessage],
        ] = {},
        context: list[SystemMessage | UserMessage | AssistantMessage] = [],
    ) -> Answer:
        untrusted_vars = dict(**inputs)

        messages = await self.fill_template(
            untrusted_inputs=untrusted_vars,
            placeholder_messages=placeholders,
            context=context,
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
            context=context,
        )
        return answer

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> Answer:
        log_message = f"{self.id} from {self.step} got request to speak."

        logger.debug(log_message)

        answer = await self.query(
            inputs=message.inputs,
            placeholders=message.placeholders,
            context=message.context,
        )

        await self.publish(answer)

        return answer
