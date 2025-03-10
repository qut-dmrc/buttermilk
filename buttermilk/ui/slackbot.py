import os
import re
from typing import Any

from pydantic import BaseModel, PrivateAttr
from slack_bolt.async_app import AsyncApp

from buttermilk.bm import logger
from buttermilk.runner.chat import ConversationManager, GroupChatMessage, IOInterface
from buttermilk.ui.slack import SlackContext

SLACK_MAX_MESSAGE_LENGTH = 3000

MODPATTERN = re.compile(
    r"^!((mod|osb|summarise_osb|trans|hate|describe|frames|simple|autogen)\s+)",
    re.IGNORECASE | re.MULTILINE,
)
ALLPATTERNS = re.compile(r"mod(.*)")

app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))


class SlackContext(BaseModel):
    channel_id: str
    thread_ts: str
    user_id: str | None = None
    say: Any = None


class MoAThread(IOInterface):
    def __init__(self, app: AsyncApp, context: SlackContext):
        self.app = app
        self.context: SlackContext = context
        self.max_message_length = 3000

    async def send_to_thread(self, text, blocks=None, **kwargs):
        kwargs.update({
            "channel": self.context.say.channel,
            "text": text,
            "blocks": blocks,
            "thread_ts": self.context.thread_ts,
        })
        msg_response = await self.context.say.client.chat_postMessage(**kwargs)
        return msg_response

    async def get_input(self, prompt: str = "") -> GroupChatMessage:
        """Retrieve input from the user interface"""
        raise NotImplementedError("MoAThread does not support get_input")

    async def send_output(self, message: GroupChatMessage, source: str = "") -> None:
        """Send output to the user interface"""
        await self.send_to_thread(text=f"{source}: {message.content}")

    async def initialize(self) -> None:
        """Initialize the interface"""
        # Start conversation
        await self.send_to_thread(text="I'm on it! Starting MoA conversation...")

    async def cleanup(self) -> None:
        """Clean up resources"""


class SlackBot(BaseModel):
    _manager: ConversationManager = PrivateAttr(
        default_factory=lambda: ConversationManager(),
    )
    flows: Any

    @app.message(MODPATTERN)
    async def moderate(self, message: dict, say):
        try:
            match = MODPATTERN.search(message["text"])
            flow_name = match[2]
            pattern_length = len(match[0])
            text = message["text"][pattern_length:]
        except Exception:
            # not formatted properly, ignore.
            return

        # Get thread id or start a new thread by getting the original ts
        thread_ts = message.get("thread_ts", message["ts"])
        channel_id = message["channel"]
        user_id = message["user"]
        event_ts = message.get("event_ts")
        client_msg_id = message.get("client_msg_id")

        # Create context for this conversation
        context = SlackContext(
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
            say=say,
        )
        # Create Slack IO interface for this thread
        io_interface = MoAThread(app, context)

        try:
            conv_id = await self._manager.start_conversation(
                io_interface=io_interface,
                platform="slack",
                external_id=f"{channel_id}-{thread_ts}",
                **self.flows[flow_name].model_dump(),
            )
            await io_interface.send_to_thread(text)

        except Exception as e:
            logger.exception(f"Error in process: {e} {e.args=}")
            await io_interface.send_to_thread(
                f"Unfortunately I hit an error, sorry: {e} {e.args=}",
            )
