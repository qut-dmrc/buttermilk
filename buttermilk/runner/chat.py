import asyncio
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

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
from buttermilk.ui.slackbot import SlackBot
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
    content: str = ""
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


class IOInterface(BaseGroupChatAgent, ABC):
    @abstractmethod
    async def get_input(self, prompt: str = "") -> GroupChatMessage:
        """Retrieve input from the user interface"""

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
        message: GroupChatMessage | Answer,
        ctx: MessageContext,
    ) -> None:
        if isinstance(message, Answer):
            source = message.agent_id
        elif ctx.sender:
            source = ctx.sender.type
        else:
            source = ctx.topic_id.type

        return await self.send_output(message, source)

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> GroupChatMessage:
        reply = await self.get_input(prompt="")
        await self.send_output(reply)
        return reply


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

    async def _run_chat(self, conv_id: ConversationId, **kwargs):
        """Run a groupchat conversation"""
        from buttermilk.runner.moa import MoA

        try:
            # Create and configure MoA
            moa = MoA(**kwargs)

            # Get the IO interface
            io = self.io_interfaces[conv_id.id]

            # Run the MoA chat with the provided IO interface
            await moa.moa_chat(io_interface=io)

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

    if False:
        # Run CLI version
        from buttermilk.ui.console import CLIUserAgent

        io_interface = CLIUserAgent
        moa = objs.flows.moa

        asyncio.run(moa.moa_chat(io_interface=io_interface))
    else:
        # Run Slack version
        slack_bot = SlackBot(flows=objs.flows)
        while True:
            asyncio.sleep(1)


if __name__ == "__main__":
    run_moa_cli()
