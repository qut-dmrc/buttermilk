import asyncio
import json
import re
from collections.abc import Mapping
from functools import partial

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from buttermilk._core.contract import (
    FlowProtocol,
)
from buttermilk._core.variants import AgentRegistry
from buttermilk.bm import logger
from buttermilk.libs.slack import SlackContext, post_message_with_retry
from buttermilk.runner.autogen import MANAGER, AutogenOrchestrator
from buttermilk.runner.chat import Selector
from buttermilk.runner.simple import Sequencer

orchestrators = [Sequencer, AutogenOrchestrator, Selector]
MODPATTERN = re.compile(
    r"^!((mod|osb|summarise_osb|trans|hate|describe|frames|simple|moa)\s+)",
    re.IGNORECASE | re.MULTILINE,
)

BOTPATTERNS = re.compile(
    r"^!?[<@>\w\d]*\s+(\w+)",
    re.IGNORECASE | re.MULTILINE,
)


ALLPATTERNS = re.compile(r"mod(.*)")


def initialize_slack_bot(
    *,
    bot_token,
    app_token,
    loop=None,
) -> tuple[AsyncApp, AsyncSocketModeHandler]:
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
    return slack_app, handler


async def register_handlers(
    slack_app: AsyncApp,
    flows: Mapping[str, FlowProtocol],
    orchestrator_tasks: asyncio.Queue,
):
    async def _flow_start_matcher(body):
        logger.debug(f"Received request: {json.dumps(body)}")
        # don't trigger on self-messages or within a thread
        return (
            body
            and body["event"].get("subtype") != "bot_message"
            and (body["event"].get("event_ts") == body["event"].get("ts"))
        )

    async def handle_mentions(body, say, logger):

        # Start conversation
        try:
            match = BOTPATTERNS.search(body["event"]["text"])
            flow_id = match[1]
            pattern_length = len(match[0])
            init_text = body["event"]["text"][pattern_length:]

        except Exception:
            # not formatted properly, ignore.
            return

        # Create context for this conversation
        context = SlackContext(
            channel_id=body["event"].get("channel"),
            thread_ts=body["event"].get("ts"),
            user_id=body["event"].get("user"),
            event_ts=body["event"].get("event_ts"),
            say=say,
        )
        # Spin up the flow in a new task so that we can get back to slack
        asyncio.create_task(
            start_flow_thread(
                context=context,
                slack_app=slack_app,
                flow_cfg=flows[flow_id],
                orchestrator_tasks=orchestrator_tasks,
                init_text=init_text,
            ),
        )

    async def say_hello(message, say):
        user = message["user"]
        await say(f"Hi there, <@{user}>!")

    slack_app.message(":wave:")(say_hello)
    slack_app.event("app_mention", matchers=[_flow_start_matcher])(handle_mentions)


async def start_flow_thread(
    context: SlackContext,
    slack_app: AsyncApp,
    flow_cfg: FlowProtocol,
    orchestrator_tasks: asyncio.Queue,
    init_text: str,
) -> None:
    logger.info("Starting autogen flow in a new slack thread...")

    from buttermilk.agents.ui.slackthreadchat import SlackUIAgent

    # Instantiate the slack thread agent before the orchestrator
    try:
        _config = dict(flow_cfg)
        orchestrator_name = _config.pop("orchestrator")
        thread_agent_name = f"slack_thread_{context.thread_ts}"
        # partially fill the SlackUIAgent object and add it to the registry
        AgentRegistry._agents[thread_agent_name] = partial(  # type: ignore
            SlackUIAgent,
            context=context,
            app=slack_app,
        )
        # and replace the fake name in the config with an identifier for this one
        _config["agents"][MANAGER]["agent_obj"] = thread_agent_name

        thread_orchestrator = globals()[orchestrator_name](**_config)

        t = asyncio.create_task(thread_orchestrator.run())
        await orchestrator_tasks.put(t)
        # # add seed message

        # await ui_config.agent_obj._input_callback(init_text)
    except Exception as e:
        logger.error(f"Error creating flow: {e}, {e.args=}")
        await post_message_with_retry(
            slack_app,
            context,
            f"Error creating flow: {e}",
        )
