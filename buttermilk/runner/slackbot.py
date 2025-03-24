import asyncio
import os
import re
from collections.abc import Mapping

from pydantic import BaseModel
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from buttermilk._core.contract import (
    FlowProtocol,
)

# Initializes app with your bot token and socket mode handler
# Starting the main slack client requires a 'xoxb-...' token
loop = asyncio.get_event_loop()
slack_app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))

MODPATTERN = re.compile(
    r"^!((mod|osb|summarise_osb|trans|hate|describe|frames|simple|moa)\s+)",
    re.IGNORECASE | re.MULTILINE,
)

ALLPATTERNS = re.compile(r"mod(.*)")


class SlackContext(BaseModel):
    channel_id: str
    thread_ts: str
    user_id: str | None = None
    event_ts: str | None = None


def initialize_slack_bot(
    *,
    bot_token,
    app_token,
    loop=None,
    flows: Mapping[str, FlowProtocol],
    orchestrator_tasks: asyncio.Queue,
) -> AsyncSocketModeHandler:
    """Initialize the Slack bot and its dependencies."""
    # Initializes app with your bot token and socket mode handler
    # Starting the main slack client requires a 'xoxb-...' token
    loop = asyncio.get_event_loop()
    slack_app = AsyncApp(token=bot_token)
    handler = AsyncSocketModeHandler(
        slack_app,
        app_token=app_token,
        loop=loop,
    )
    handler.app.message(":wave:")(say_hello)

    async def handle_mentions(body, say, logger):
        logger.info("Starting autogen flow in a new slack thread...")
        from buttermilk.agents.ui.slackthreadchat import start_flow_thread

        await start_flow_thread(
            body,
            app=slack_app,
            flows=flows,
            orchestrator_tasks=orchestrator_tasks,
        )

    return handler


async def say_hello(message, say):
    user = message["user"]
    await say(f"Hi there, <@{user}>!")
