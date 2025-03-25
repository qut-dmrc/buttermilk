from typing import Any

from pydantic import PrivateAttr

from buttermilk._core.contract import (
    AgentMessages,
    AgentOutput,
    ManagerMessage,
    UserConfirm,
    UserInput,
)
from buttermilk.agents.ui.formatting.slackblock import (
    confirm_block,
    format_slack_message,
)
from buttermilk.agents.ui.generic import UIAgent
from buttermilk.libs.slack import SlackContext, post_message_with_retry


class SlackUIAgent(UIAgent):
    # these need to be populated after the agent is created by the factory
    app: Any = None
    context: "SlackContext" = None
    _input_callback: Any = PrivateAttr(default=None)

    async def send_to_thread(self, text, blocks=None, **kwargs):
        return await post_message_with_retry(
            app=self.app,
            context=self.context,
            text=text,
            blocks=blocks,
            **kwargs,
        )

    async def receive_output(
        self,
        message: AgentOutput | ManagerMessage | UserConfirm | UserInput,
        source: str,
        **kwargs,
    ) -> None:
        """Send output to the Slack thread"""
        if isinstance(message, (UserInput, UserConfirm)):
            return
        if isinstance(message, AgentOutput):
            try:
                formatted_blocks = format_slack_message(message)
                await self.send_to_thread(**formatted_blocks)
            except:
                await self.send_to_thread(text=message.content)
        else:
            await self.send_to_thread(text=message.content)

    async def confirm(self, message: str = ""):
        confirm_blocks = confirm_block(
            message=message or "Would you like to proceed?",
        )
        await self.send_to_thread(
            text=confirm_blocks["text"],
            blocks=confirm_blocks["blocks"],
        )

    async def _process(
        self,
        input_data: AgentMessages,
        **kwargs,
    ) -> AgentOutput | None:
        """Handle input requests including confirmations"""
        # For confirmation requests, display buttons
        if not input_data.content or "confirm" in input_data.content.lower():
            return await self.confirm(input_data.content)
        # For regular prompts, just display the message
        await self.send_to_thread(text=input_data.content)

        return None  # We'll handle responses via the callback system

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface and register handlers"""
        self._input_callback = input_callback

        await self.send_to_thread(text="I'm on it! Starting conversation...")
        await self.confirm()

        # Register this agent's thread for message handling
        register_chat_thread_handler(self.context.thread_ts, self)

    async def cleanup(self) -> None:
        """Clean up resources and unregister handlers"""
        # Could add logic to unregister handlers if needed


def register_chat_thread_handler(thread_ts, agent: SlackUIAgent):
    """Connect messages in a Slack thread to the agent's callback"""

    async def matcher(message):
        return (
            # It's a message in our thread, not from us.
            message.get("thread_ts") == thread_ts
            and message.get("subtype") != "bot_message"
        )

    @agent.app.message(matchers=[matcher])
    async def feed_in(message, say):
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
