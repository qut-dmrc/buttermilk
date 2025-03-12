import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Union

import hydra
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
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from buttermilk._core.agent import AgentConfig
from buttermilk._core.runner_types import RecordInfo
from buttermilk.bm import logger
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    KeyValueCollector,
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


class NullAnswer(GroupChatMessage):
    type: str = "NullAnswer"
    content: str = ""
    """A message sent to the group chat indicating that the agent did not provide an answer."""


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


# Union of all known GroupChatMessage subclasses
GroupChatMessageType = Union[GroupChatMessage, NullAnswer, InputRecord, Answer]


class RequestToSpeak(BaseModel):
    content: str | None = None
    inputs: Mapping[str, Any] = {}
    placeholders: Mapping[
        str,
        list[SystemMessage | UserMessage | AssistantMessage],
    ] = {}
    context: list[SystemMessage | UserMessage | AssistantMessage] = []


class MessagesCollector(KeyValueCollector):
    """Specifically typed to collect pairs of (User, Assistant) messages"""

    _data: dict[str, list[UserMessage | AssistantMessage]] = PrivateAttr(
        default_factory=dict,
    )


class BaseGroupChatAgent(RoutedAgent, ABC):
    """A group chat participant."""

    def __init__(
        self,
        config: AgentConfig,
        group_chat_topic_type: str = "default",
    ) -> None:
        """Initialize the agent with configuration and topic type.

        Args:
            config: Configuration settings for the agent
            group_chat_topic_type: The type of group chat topic to use (default: "default")

        """
        super().__init__(
            description=config.description,
        )
        self.config = config
        self.step = config.name
        self.parameters = config.parameters
        self._group_chat_topic_type = group_chat_topic_type
        self._json_parser = ChatParser()

    async def publish(self, message: Any) -> None:
        await self.publish_message(
            message,
            DefaultTopicId(type=self._group_chat_topic_type),
        )

    @abstractmethod
    async def query(self, request: RequestToSpeak) -> GroupChatMessageType:
        """Query the agent with the given inputs and placeholders."""
        raise NotImplementedError

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> GroupChatMessageType:
        log_message = f"{self.id} got request to speak."

        logger.debug(log_message)

        answer = await self.query(message)

        await self.publish(answer)

        return answer


class IOInterface(BaseGroupChatAgent, ABC):
    def __init__(
        self,
        group_chat_topic_type: str,
    ) -> None:
        ui_config = AgentConfig(
            agent=self.__class__.__name__,
            name="user",
            description="User interface",
        )
        super().__init__(ui_config, group_chat_topic_type)

    @abstractmethod
    async def send_output(self, message: GroupChatMessage, source: str = "") -> None:
        """Send output to the user interface"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the interface"""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""

    @message_handler
    async def handle_message(
        self,
        message: GroupChatMessageType,
        ctx: MessageContext,
    ) -> None:
        if isinstance(message, Answer):
            source = message.agent_id
        elif ctx.sender:
            source = ctx.sender.type
        else:
            source = ctx.topic_id.type

        await self.send_output(message, source)


class ConversationId(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    platform: str
    external_id: str


class ConversationManager(BaseModel):
    conversations: dict[str, Any] = {}
    io_interfaces: dict[str, Any] = {}

    async def start_conversation(
        self,
        io_interface: IOInterface,
        platform: str,
        external_id: str,
        init_text: str = None,
        history: list = [],
        **kwargs,
    ) -> ConversationId:
        """Start a new group chat conversation with the given IO interface"""
        conv_id = ConversationId(platform=platform, external_id=external_id)

        # Store the IO interface
        self.io_interfaces[conv_id.id] = io_interface

        # Create and start the MoA task
        task = asyncio.create_task(self._run_chat(conv_id, **kwargs))
        self.conversations[conv_id.id] = task

        return conv_id

    async def _run_chat(self, conv_id: ConversationId, init_text: str = None, **kwargs):
        """Run a groupchat conversation"""
        from buttermilk.runner.moa import MoA

        try:
            # Create and configure MoA
            moa = MoA(**kwargs)

            # Get the IO interface
            io = self.io_interfaces[conv_id.id]

            # Run the MoA chat with the provided IO interface
            await moa.moa_chat(io_interface=io, init_text=init_text)

        except Exception as e:
            logger.exception(f"Error in conversation {conv_id.id}: {e!s}")
        finally:
            # Clean up
            if conv_id.id in self.io_interfaces:
                await self.io_interfaces[conv_id.id].cleanup()
                del self.io_interfaces[conv_id.id]

            if conv_id.id in self.conversations:
                del self.conversations[conv_id.id]


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def run_moa_cli(cfg) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm

    if bm.cfg.run.ui == "cli":
        # Run CLI version
        from buttermilk.ui.console import CLIUserAgent

        moa = objs.flows[cfg.flow]
        # flow = objs.flows[cfg.flow]
        # moa = MoA(steps=flow.steps, source="dev")
        asyncio.run(moa.moa_chat(io_interface=CLIUserAgent))
    elif bm.cfg.run.ui == "slack":
        # Run Slack version

        secrets = bm.secret_manager.get_secret("automod")
        os.environ["SLACK_BOT_TOKEN"] = secrets["AUTOMOD_BOT_TOKEN"]
        app_token = secrets["DMRC_SLACK_APP_TOKEN"]
        manager = ConversationManager()
        flows = objs.flows
        from buttermilk.ui.slackbot import initialize_slack_bot

        loop = asyncio.get_event_loop()
        handler = initialize_slack_bot(
            conversation_manager=manager,
            flows=flows,
            loop=loop,
            app_token=app_token,
        )
        loop.run_until_complete(handler.start_async())
    else:
        raise ValueError(f"Unknown run ui type: {bm.cfg.run.ui}")

    pass  # noqa


if __name__ == "__main__":
    run_moa_cli()
