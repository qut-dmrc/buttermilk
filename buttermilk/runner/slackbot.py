import asyncio
import json
import re
import traceback
from collections.abc import Mapping
from functools import partial
from typing import Any

from autogen_core.models import AssistantMessage, UserMessage
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from buttermilk._core.orchestrator import OrchestratorProtocol
from buttermilk._core.types import RunRequest
from buttermilk._core.variants import AgentRegistry
from buttermilk.bm import BM, logger  # Buttermilk global instance and logger

bm = BM()
from buttermilk.libs.slack import SlackContext, post_message_with_retry
from buttermilk.orchestrators.groupchat import AutogenOrchestrator  #noqa

_ = "AutogenOrchestrator"
BOTPATTERNS = re.compile(
    r"^!?[<@>\w\d]*\s+(\w+)(.*)",
    re.IGNORECASE | re.MULTILINE,
)
RESUME = "resume"

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

    # Add reconnection handler
    from buttermilk.agents.ui.slackthreadchat import reregister_all_active_threads

    @slack_app.event("hello")
    async def on_connect():
        logger.info("Socket Mode client reconnected")
        # Re-register handlers for all active threads
        reregister_all_active_threads()

    return slack_app, handler


async def register_handlers(
    slack_app: AsyncApp,
    flows: Mapping[str, OrchestratorProtocol],
    orchestrator_tasks: asyncio.Queue,
):
    async def _flow_start_matcher(body):
        logger.debug(f"Received request: {json.dumps(body)}")
        # don't trigger on self-messages or within a thread
        if body and body["event"].get("subtype") != "bot_message" and (body["event"].get("event_ts") != body["event"].get("thread_ts")):
            match = BOTPATTERNS.search(body["event"]["text"])
            if match:
                return True

        # trigger on a 'resume' message in a thread
        if (
            body
            and body["event"].get("subtype") != "bot_message"
            and body["event"].get("thread_ts")
            and (body["event"].get("event_ts") == body["event"].get("thread_ts"))
        ):
            if match := BOTPATTERNS.search(body["event"]["text"]):
                if str.lower(match[1]) == RESUME:
                    return True
        return False

    async def handle_mentions(body, say, logger):
        await say.client.reactions_add(
            channel=say.channel,
            name="eyes",
            timestamp=body["event"]["ts"],
        )
        try:
            init_text = ""
            # Start conversation
            flow_id = None
            if match := BOTPATTERNS.search(body["event"]["text"]):
                flow_id = match[1]
                if str.lower(match[1]) == RESUME:
                    text = match[2].split(maxsplit=1)
                    flow_id = text[0]
                    init_text = text[1] if len(text) > 1 else ""
                else:
                    init_text = match[2]

                logger.info(
                    "Received flow request",
                    extra={
                        "flow_id": flow_id,
                        "channel_id": body["event"].get("channel"),
                        "user_id": body["event"].get("user"),
                        "thread_ts": body["event"].get(
                            "thread_ts",
                            body["event"].get("ts"),
                        ),
                    },
                )

                if flow_id not in flows:
                    logger.warning(
                        "Invalid flow requested",
                        extra={
                            "flow_id": flow_id,
                            "available_flows": list(flows.keys()),
                            "user_id": body["event"].get("user"),
                        },
                    )
                    await post_message_with_retry(
                        slack_app,
                        context=SlackContext(
                            channel_id=body["event"].get("channel"),
                            thread_ts=body["event"].get(
                                "thread_ts",
                                body["event"].get("ts"),
                            ),
                            user_id=body["event"].get("user"),
                            event_ts=body["event"].get("event_ts"),
                            say=say,
                        ),
                        text=f"I don't know how to run flow '{flow_id}'. Available flows: {', '.join(flows.keys())}",
                    )
                    return

            # Create context for this conversation
            context = SlackContext(
                channel_id=body["event"].get("channel"),
                thread_ts=body["event"].get("thread_ts", body["event"].get("ts")),
                user_id=body["event"].get("user"),
                event_ts=body["event"].get("event_ts"),
                say=say,
            )
            # Spin up the flow in a new task so that we can get back to slack
            logger.info(
                "Starting flow task",
                extra={
                    "flow_id": flow_id,
                    "channel_id": context.channel_id,
                    "thread_ts": context.thread_ts,
                    "user_id": context.user_id,
                },
            )
            t = asyncio.create_task(
                start_flow_thread(
                    bm=bm,
                    context=context,
                    slack_app=slack_app,
                    flow_cfg=flows[flow_id],
                    orchestrator_tasks=orchestrator_tasks,
                    init_text=init_text,
                ),
            )
        except Exception as e:
            logger.exception(
                "Error handling mention",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "body": body,
                    "traceback": traceback.format_exc(),
                },
            )
            # Try to notify the user if we can
            try:
                await say(f"Error processing your request: {e}")
            except Exception:
                pass

    async def say_hello(message, say):
        user = message["user"]
        await say(f"Hi there, <@{user}>!")

    slack_app.message(":wave:")(say_hello)
    slack_app.event("app_mention", matchers=[_flow_start_matcher])(handle_mentions)

    def _handle_unmatched_requests(req: Any, resp: Any) -> Any:
        # Just silently acknowledge the message without doing anything
        # Only log at debug level to avoid cluttering logs
        logger.debug("Received unhandled message event")
        # by default slack sends a warning:
        # self._framework_logger.warning(warning_unhandled_request(req))
        return resp

    slack_app._handle_unmatched_requests = _handle_unmatched_requests


async def read_thread_history(
    slack_app: AsyncApp,
    context: SlackContext,
) -> list[dict[str, str]]:
    # Read thread history
    replies = await slack_app.client.conversations_replies(
        channel=context.channel_id,
        ts=context.thread_ts,
    )
    history = []
    if replies and "messages" in replies:
        for message in replies["messages"]:
            if message.get("text", "").startswith("<@") or message.get("text", "").startswith("!"):
                # ignore directed messages
                continue
            user = message.get("user", "Unknown")
            text = message.get("text", "")
            if message.get("subtype") == "bot_message":
                history.append(AssistantMessage(content=text, source="slack-thread"))
            else:
                history.append(UserMessage(content=text, source="slack-thread"))

    return history


async def start_flow_thread(
    bm: BM,
    context: SlackContext,
    slack_app: AsyncApp,
    flow_cfg: OrchestratorProtocol,
    orchestrator_tasks: asyncio.Queue,
    init_text: str,
) -> None:
    logger.info(
        f"Starting flow {flow_cfg.name} in Slack thread {context.thread_ts}...",
        extra={
            "flow_name": flow_cfg.name,
            "channel_id": context.channel_id,
            "thread_ts": context.thread_ts,
            "user_id": context.user_id,
        },
    )

    from buttermilk.agents.ui.slackthreadchat import SlackUIAgent

    # Instantiate the slack thread agent before the orchestrator
    try:
        _config = flow_cfg  # OmegaConf.to_container(flow_cfg, resolve=True)
        orchestrator_name = _config.orchestrator.split(".")[-1]
        thread_agent_name = f"slack_thread_{context.thread_ts}"
        # partially fill the SlackUIAgent object and add it to the registry
        AgentRegistry._agents[thread_agent_name] = partial(  # type: ignore
            SlackUIAgent,
            context=context,
            app=slack_app,
        )
        # and replace the fake name in the config with an identifier for this one
        _config.observers["manager"].agent_obj = thread_agent_name

        # Read thread history and append init text
        logger.debug(
            "Fetching thread history",
            extra={
                "thread_ts": context.thread_ts,
                "channel_id": context.channel_id,
            },
        )
        history = await read_thread_history(slack_app=slack_app, context=context)
        # Remove first entry (activation message)
        if history:
            history.pop(0)

        # Prepend init_text if it's not empty
        if init_text.strip():
            history.append(UserMessage(content=init_text, source="slack-thread"))
            logger.debug(
                "Added initial text to history",
                extra={"text_length": len(init_text)},
            )

        logger.info(
            f"Creating {orchestrator_name} orchestrator",
        )
        thread_orchestrator = globals()[orchestrator_name](**_config.model_dump())

        t = asyncio.create_task(thread_orchestrator.run(RunRequest(flow=flow_cfg.name, ui_type="slack-thread")))
        await orchestrator_tasks.put(t)
        logger.debug(
            "Flow task created and queued",
        )
    except Exception as e:
        logger.error(
            f"Error creating flow: {e} {e.args=}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "flow_name": flow_cfg.name,
                "thread_ts": context.thread_ts,
                "traceback": traceback.format_exc(),
            },
        )
        await post_message_with_retry(
            slack_app,
            context,
            f"Error creating flow: {e} {e.args=}",
        )
