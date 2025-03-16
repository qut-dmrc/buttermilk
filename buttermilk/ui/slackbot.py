import os
import re
from asyncio import Queue

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
from buttermilk.ui.formatting.slackblock import format_response
from buttermilk.ui.formatting.slackblock_reasons import format_slack_reasons

from tenacity import (
    retry,
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)
import asyncio

SLACK_MAX_MESSAGE_LENGTH = 3000

MODPATTERN = re.compile(
    r"^!((mod|osb|summarise_osb|trans|hate|describe|frames|simple|moa)\s+)",
    re.IGNORECASE | re.MULTILINE,
)
BOTPATTERNS = re.compile(
    r"^!?[<@>\w\d]*\s+(\w+)",
    re.IGNORECASE | re.MULTILINE,
)

ALLPATTERNS = re.compile(r"mod(.*)")

# Global state variables
app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))
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
        event_ts: str | None = None

    class MoAThreadNoContext(IOInterface):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.app = app
            self.context: SlackContext = None
            self.max_message_length = 3000
            self.input_queue = Queue()

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10)
        )
        async def _post_message_with_retry(self, **kwargs):
            return await app.client.chat_postMessage(**kwargs)

        async def send_to_thread(self, text, blocks=None, **kwargs):
            kwargs.update({
                "channel": self.context.channel_id,
                "text": text,
                "blocks": blocks,
                "thread_ts": self.context.thread_ts,
            })
            
            msg_response = await self._post_message_with_retry(**kwargs)
            return msg_response
        
        async def query(self, request: RequestToSpeak) -> GroupChatMessage:
            """Retrieve input from the user interface"""
            if request.content:
                await self.send_to_thread(request.content)
            else:
                from buttermilk.ui.formatting.slackblock import confirm_block
                confirm_blocks = confirm_block(message=request.prompt or "Would you like to proceed?")
                response = await self.send_to_thread(text=confirm_blocks["text"], blocks=confirm_blocks["blocks"])
                
                # Setup action handlers for the buttons
                @app.action("confirm_action")
                async def handle_confirm(ack, body, client):
                    await ack()
                    user_id = body["user"]["id"]
                    # Put "Yes" response in the queue
                    await self.input_queue.put(True)
                    # Replace buttons with confirmation message
                    await client.chat_update(
                        channel=self.context.channel_id,
                        ts=response["ts"],
                        text="You selected: Yes",
                        blocks=[{
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": ":white_check_mark: You selected: *Yes*"
                            }
                        }]
                    )
                    
                @app.action("cancel_action")
                async def handle_cancel(ack, body, client):
                    await ack()
                    user_id = body["user"]["id"]
                    # Put "No" response in the queue
                    await self.input_queue.put(False)
                    # Replace buttons with cancellation message
                    await client.chat_update(
                        channel=self.context.channel_id,
                        ts=response["ts"],
                        text="You selected: No",
                        blocks=[{
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": ":x: You selected: *No*"
                            }
                        }]
                    )
                    
            # Wait for a response (either text message or button click)
            msg = await self.input_queue.get()

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
            if isinstance(message, Answer):        
                formatted_blocks = format_slack_reasons(message)
                await self.send_to_thread(**formatted_blocks)
            else:
                await self.send_to_thread(message.content)

        async def initialize(self) -> None:
            """Initialize the interface"""
            # Start conversation
            await self.send_to_thread(text="I'm on it! Starting MoA conversation...")

        async def cleanup(self) -> None:
            """Clean up resources"""

    @app.event("app_mention")
    async def handle_app_mention_events(body, logger):
        logger.info(body)
        return await run_moderate(body["event"])

    @app.event("message_replied")
    async def handle_thread_reply_resume(message, logger):
        try:
            match = BOTPATTERNS.search(message["text"])
            flow_name = match[1]
            pattern_length = len(match[0])
            step = message["text"][pattern_length:]

            flow = _flows[flow_name]
            history = await app.client.channels_replies(
                message["channel"], message["message"]["thread_ts"]
            )

            return  # not implemented yet
        except Exception:
            # not formatted properly, ignore.
            return

    @app.message(MODPATTERN)
    async def handle_keyword(message: dict, say):
        return await run_moderate(message)

    async def run_moderate(message: dict):
        try:
            match = BOTPATTERNS.search(message["text"])
            flow_name = match[1]
            pattern_length = len(match[0])
            init_text = message["text"][pattern_length:]
            flow = _flows[flow_name]
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
            event_ts=message.get("event_ts"),
        )

        await moderate(context=context, flow=flow)

    async def moderate(context, flow, init_text=None, history=[]):
        class SlackMoAThread(MoAThreadNoContext):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.context = context

                # communication from slack to chat thread
                register_chat_thread_handler(self.context.thread_ts, self)

        # Create Slack IO interface for this thread
        io_interface = SlackMoAThread

        try:
            await _manager.start_conversation(
                io_interface=io_interface,
                init_text=init_text,
                platform="slack",
                external_id=f"{context.channel_id}-{context.thread_ts}",
                **flow.model_dump(),
            )

        except Exception as e:
            logger.exception(f"Error in process: {e} {e.args=}")
            await io_interface.send_to_thread(
                f"Unfortunately I hit an error, sorry: {e} {e.args=}",
            )


def register_chat_thread_handler(thread_ts, agent: "MoAThreadNoContext"):
    """Connect messages sent to a slack thread to the group chat."""

    async def matcher(message):
        return (
            message.get("thread_ts") == thread_ts
            and message.get("subtype") != "bot_message"
        )

    @app.message(
        matchers=[matcher],
    )
    async def feed_in(message, say):
        await agent.input_queue.put(message["text"])


def initialize_slack_bot(
    *,
    conversation_manager,
    flows,
    loop,
    app_token,
) -> AsyncSocketModeHandler:
    """Initialize the Slack bot and its dependencies."""
    global _flows
    _flows = flows

    # Register all handlers
    register_handlers()

    handler = AsyncSocketModeHandler(app, app_token=app_token, loop=loop)
    return handler
