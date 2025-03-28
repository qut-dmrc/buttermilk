from typing import Any

import pydantic
from pydantic import PrivateAttr
from rich.console import Console
from rich.markdown import Markdown
from slack_bolt.async_app import AsyncApp

from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentMessages,
    AgentOutput,
    ManagerRequest,
    UserInstructions,
)
from buttermilk.agents.ui.formatting.slackblock import (
    confirm_bool,
    confirm_options,
    dict_to_blocks,
    format_slack_message,
)
from buttermilk.agents.ui.generic import UIAgent
from buttermilk.libs.slack import SlackContext, post_message_with_retry


def _fn_debug_blocks(message: AgentOutput):
    return
    try:
        console = Console(highlight=True)
        console.print(Markdown("## -----DEBUG BLOCKS------"))
        console.print_json(data=format_slack_message(message.outputs))
        console.print(Markdown("## -----DEBUG BLOCKS------"))
    except:
        pass


class SlackUIAgent(UIAgent):
    # these need to be populated after the agent is created by the factory
    app: AsyncApp = None
    context: "SlackContext" = None
    _input_callback: Any = PrivateAttr(default=None)
    _current_input_message: Any = PrivateAttr(default=None)
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    async def send_to_thread(self, text=None, blocks=None, **kwargs):
        return await post_message_with_retry(
            app=self.app,
            context=self.context,
            text=text,
            blocks=blocks,
            **kwargs,
        )

    async def receive_output(
        self,
        message: AgentMessages | UserInstructions,
        **kwargs,
    ) -> None:
        """Send output to the Slack thread"""
        if isinstance(message, AgentOutput | AgentInput):
            try:
                formatted_blocks = format_slack_message(message)
                await self.send_to_thread(**formatted_blocks)
            except Exception as e:  # noqa
                _fn_debug_blocks(message)
                await self.send_to_thread(text=message.content)

    async def _request_user_input(self, message: ManagerRequest, **kwargs) -> None:
        """Ask for user input from the UI."""
        if isinstance(message, (AgentInput, ManagerRequest)):
            extra_blocks = dict_to_blocks(message.inputs)
            if isinstance(message, ManagerRequest) and message.options is not None:
                if isinstance(message.options, bool):
                    # If there are binary options, display buttons
                    confirm_blocks = confirm_bool(
                        message=message.content,
                        extra_blocks=extra_blocks,
                    )
                elif isinstance(message.options, list):
                    # If there are multiple options, display a dropdown
                    confirm_blocks = confirm_options(
                        message=message.content,
                        options=message.options,
                        extra_blocks=extra_blocks,
                    )
                else:
                    raise ValueError("Invalid options type")
            else:
                # Assume binary yes / no
                confirm_blocks = confirm_bool(
                    message=message.content,
                    extra_blocks=extra_blocks,
                )

            if self._current_input_message is not None:
                try:
                    # we need to update the current message instead of
                    # opening a new one.
                    await self.app.client.chat_update(
                        channel=self.context.channel_id,
                        ts=self._current_input_message.data["ts"],
                        text=confirm_blocks["text"],
                        blocks=confirm_blocks["blocks"],
                    )
                except:
                    pass
            # We don't have an open input message. Send a new one.
            self._current_input_message = await self.send_to_thread(
                text=confirm_blocks["text"],
                blocks=confirm_blocks["blocks"],
            )
        else:
            raise ValueError("Invalid message type")

    async def _process(
        self,
        input_data: AgentMessages,
        **kwargs,
    ) -> AgentOutput | None:
        """Tell the user we're expecting some data, but don't wait around"""
        await self._request_user_input(input_data)

        return None  # We'll handle responses via the callback system

    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface and register handlers"""
        self._input_callback = input_callback

        # Register this agent's thread for message handling
        register_chat_thread_handler(self.context.thread_ts, self)

    async def cleanup(self) -> None:
        """Clean up resources and unregister handlers"""
        # Could add logic to unregister handlers if needed


def register_chat_thread_handler(thread_ts, agent: SlackUIAgent):
    """Connect messages in a Slack thread to the agent's callback"""

    async def matcher(message):
        return (
            # It's a message in our thread, not from the bot.
            message.get("thread_ts") == thread_ts
            and message.get("subtype") != "bot_message"
        )

    @agent.app.message(matchers=[matcher])
    async def feed_in(message, say):
        await agent._input_callback(UserInstructions(content=message["text"]))

    # Button action handlers
    @agent.app.action("confirm_action")
    async def handle_confirm(ack, body, client):
        await ack()

        if not body.get("message", {}).get("thread_ts") == thread_ts:
            # why are we here?
            logger.debug("Received message not for us.")
            return

        # Only proceed if we are sure this message has not been dealt with yet.
        if agent._current_input_message is None:
            return

        # Clear reference to input message
        agent._current_input_message = None

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
        await agent._input_callback(UserInstructions(confirm=True))
        agent._current_input_message = None

    @agent.app.action("cancel_action")
    async def handle_cancel(ack, body, client):
        await ack()
        if not body.get("message", {}).get("thread_ts") == thread_ts:
            # why are we here?
            logger.debug("Received message not for us.")
            return

        # Only proceed if we are sure this message has not been dealt with yet.
        if agent._current_input_message is None:
            return

        # Clear reference to input message
        agent._current_input_message = None

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
        await agent._input_callback(UserInstructions(confirm=False))
