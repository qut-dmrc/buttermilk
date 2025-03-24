import asyncio
import re
from typing import Any

from pydantic import PrivateAttr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from buttermilk._core.contract import (
    AgentMessages,
    AgentOutput,
)
from buttermilk.agents.ui.formatting.slackblock import confirm_block
from buttermilk.agents.ui.formatting.slackblock_reasons import format_slack_reasons
from buttermilk.agents.ui.generic import UIAgent
from buttermilk.runner.slackbot import SlackContext

BOTPATTERNS = re.compile(
    r"^!?[<@>\w\d]*\s+(\w+)",
    re.IGNORECASE | re.MULTILINE,
)


class SlackUIAgent(UIAgent):
    app: Any
    context: "SlackContext"
    _input_callback: Any = PrivateAttr(default=None)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _post_message_with_retry(self, **kwargs):
        return await self.app.client.chat_postMessage(**kwargs)

    async def send_to_thread(self, text, blocks=None, **kwargs):
        kwargs.update({
            "channel": self.context.channel_id,
            "text": text,
            "blocks": blocks,
            "thread_ts": self.context.thread_ts,
        })
        return await self._post_message_with_retry(**kwargs)

    async def receive_output(
        self,
        message: AgentMessages,
        source: str,
        **kwargs,
    ) -> None:
        """Send output to the Slack thread"""
        if hasattr(message, "reasons") and message.reasons:
            formatted_blocks = format_slack_reasons(message)
            await self.send_to_thread(**formatted_blocks)
        else:
            await self.send_to_thread(text=message.content)

    async def process(
        self,
        input_data: AgentMessages,
        **kwargs,
    ) -> AgentOutput | None:
        """Handle input requests including confirmations"""
        # For confirmation requests, display buttons
        if not input_data.content or "confirm" in input_data.content.lower():
            confirm_blocks = confirm_block(
                message=input_data.content or "Would you like to proceed?",
            )
            await self.send_to_thread(
                text=confirm_blocks["text"],
                blocks=confirm_blocks["blocks"],
            )
        else:
            # For regular prompts, just display the message
            await self.send_to_thread(text=input_data.content)

        return None  # We'll handle responses via the callback system

    async def initialize(self, **kwargs) -> None:
        """Initialize the interface and register handlers"""
        if "input_callback" in kwargs:
            self._input_callback = kwargs["input_callback"]

        await self.send_to_thread(text="I'm on it! Starting conversation...")

        # Register this agent's thread for message handling
        register_chat_thread_handler(self.context.thread_ts, self)

    async def cleanup(self) -> None:
        """Clean up resources and unregister handlers"""
        # Could add logic to unregister handlers if needed


def register_chat_thread_handler(thread_ts, agent: SlackUIAgent):
    """Connect messages in a Slack thread to the agent's callback"""

    async def matcher(message):
        return (
            message.get("thread_ts") == thread_ts
            and message.get("subtype") != "bot_message"
        )

    @agent.app.message(matchers=[matcher])
    async def feed_in(message, say):
        if hasattr(agent, "_input_callback") and agent._input_callback:
            await agent._input_callback(message["text"])

    # Button action handlers
    @agent.app.action("confirm_action")
    async def handle_confirm(ack, body, client):
        if body.get("message", {}).get("thread_ts") == thread_ts:
            await ack()
            # Update UI to show confirmation
            await client.chat_update(
                channel=agent.context.channel_id,
                ts=body["message"]["ts"],
                text="You selected: Yes",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":white_check_mark: You selected: *Yes*",
                        },
                    },
                ],
            )
            # Call callback with boolean True
            if hasattr(agent, "_input_callback") and agent._input_callback:
                await agent._input_callback("True")

    @agent.app.action("cancel_action")
    async def handle_cancel(ack, body, client):
        if body.get("message", {}).get("thread_ts") == thread_ts:
            await ack()
            # Update UI to show cancellation
            await client.chat_update(
                channel=agent.context.channel_id,
                ts=body["message"]["ts"],
                text="You selected: No",
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": ":x: You selected: *No*"},
                    },
                ],
            )
            # Call callback with boolean False
            if hasattr(agent, "_input_callback") and agent._input_callback:
                await agent._input_callback("False")


async def start_flow_thread(body, app, flows, orchestrator_tasks) -> None:
    # TODO: don't trigger within a thread

    # Start conversation
    try:
        match = BOTPATTERNS.search(body["text"])
        flow_name = match[1]
        pattern_length = len(match[0])
        init_text = body["text"][pattern_length:]

    except Exception:
        # not formatted properly, ignore.
        return

    # Create context for this conversation
    context = SlackContext(
        channel_id=body["channel"],
        thread_ts=body.get("thread_ts", body["ts"]),
        user_id=body["user"],
        event_ts=body.get("event_ts"),
    )
    flow_cfg = flows[flow_name]
    orchestrator_name = flow_cfg.orchestrator

    # Instantiate the slack thread agent before the orchestrator
    ui_config = flow_cfg.steps[0]
    ui_config.agent_obj = SlackUIAgent(
        context=context,
        app=app,
        **ui_config,
    )
    thread_orchestrator = globals()[orchestrator_name](**flow_cfg)
    t = asyncio.create_task(thread_orchestrator.run())
    await orchestrator_tasks.put(t)

    return
