import os
import re
from asyncio import Queue
from typing import Any

from pydantic import BaseModel
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from buttermilk.bm import logger
from buttermilk.runner.chat import (
    Answer,
    ConversationManager,
    GroupChatMessage,
    IOInterface,
    RequestToSpeak,
)

SLACK_MAX_MESSAGE_LENGTH = 3000

MODPATTERN = re.compile(
    r"^!((mod|osb|summarise_osb|trans|hate|describe|frames|simple|moa)\s+)",
    re.IGNORECASE | re.MULTILINE,
)
ALLPATTERNS = re.compile(r"mod(.*)")

# Global state variables
app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))
_conversation_manager = None
_flows = None
_manager = ConversationManager()


def register_handlers():
    """Register all event handlers."""

    @app.message(":wave:")
    async def say_hello(message, say):
        user = message["user"]
        await say(f"Hi there, <@{user}>!")

    class SlackContext(BaseModel):
        channel_id: str
        thread_ts: str
        user_id: str | None = None
        say: Any = None
        event_ts: str | None = None
        client_msg_id: str | None = None

    class MoAThreadNoContext(IOInterface):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.app = app
            self.context: SlackContext = None
            self.max_message_length = 3000
            self.inputqueue = Queue()

        async def send_to_thread(self, text, blocks=None, **kwargs):
            kwargs.update({
                "channel": self.context.say.channel,
                "text": text,
                "blocks": blocks,
                "thread_ts": self.context.thread_ts,
            })
            msg_response = await self.context.say.client.chat_postMessage(**kwargs)
            return msg_response

        async def query(self, request: RequestToSpeak) -> GroupChatMessage:
            """Retrieve input from the user interface"""
            await self.send_to_thread(request.content or "Enter your message: ")
            msg = await self.inputqueue.get()

            reply = Answer(
                agent_id=self.id.type,
                role="user",
                content=msg,
                step="User",
                config=self.config,
            )
            return reply

        async def send_output(
            self,
            message: GroupChatMessage,
            source: str = "",
        ) -> None:
            """Send output to the user interface"""
            await self.send_to_thread(text=f"{source}: {message.content}")

        async def initialize(self) -> None:
            """Initialize the interface"""
            # Start conversation
            await self.send_to_thread(text="I'm on it! Starting MoA conversation...")

        async def cleanup(self) -> None:
            """Clean up resources"""

    @app.message(MODPATTERN)
    async def moderate(message: dict, say):
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

        # Create context for this conversation
        context = SlackContext(
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
            say=say,
            event_ts=message.get("event_ts"),
            client_msg_id=message.get("client_msg_id"),
        )

        class MoAThread(MoAThreadNoContext):
            def __init__(self, **kwargs):
                super().__init__(init_text=text, **kwargs)
                self.context = context

                # communication from slack to chat thread
                register_chat_thread_handler(thread_ts, self)

        # Create Slack IO interface for this thread
        io_interface = MoAThread

        try:
            await _manager.start_conversation(
                io_interface=io_interface,
                platform="slack",
                external_id=f"{channel_id}-{thread_ts}",
                **_flows[flow_name].model_dump(),
            )

        except Exception as e:
            logger.exception(f"Error in process: {e} {e.args=}")
            await io_interface.send_to_thread(
                f"Unfortunately I hit an error, sorry: {e} {e.args=}",
            )


def register_chat_thread_handler(thread_ts, agent: "MoAThreadNoContext"):
    """Connect messages sent to a slack thread to the group chat."""

    @app.message(
        matchers=lambda message: message.thread_ts == thread_ts
        and message.get("subtype") != "bot_message",
    )
    async def feed_in(self, message, say):
        await agent.input_queue.put(message.text)


def initialize_slack_bot(
    *,
    conversation_manager,
    flows,
    loop,
    app_token,
) -> AsyncSocketModeHandler:
    """Initialize the Slack bot and its dependencies."""
    global _conversation_manager, _flows
    _conversation_manager = conversation_manager
    _flows = flows

    # Register all handlers
    register_handlers()

    handler = AsyncSocketModeHandler(app, app_token=app_token, loop=loop)
    return handler
