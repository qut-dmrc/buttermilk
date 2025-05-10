from collections.abc import Callable
from typing import Any

import pydantic
from autogen_core import CancellationToken
from pydantic import BaseModel, PrivateAttr
from rich.console import Console
from rich.markdown import Markdown
from slack_bolt.async_app import AsyncApp

from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
    OOBMessages,
)
from buttermilk.agents.ui.formatting.slackblock import (
    confirm_bool,
    confirm_options,
    dict_to_blocks,
    format_slack_message,
)
from buttermilk.agents.ui.generic import UIAgent
from buttermilk.libs.slack import (
    SlackContext,
    post_message_with_retry,
)

_active_thread_registry = {}


def _fn_debug_blocks(message: AgentTrace):
    try:
        console = Console(highlight=True)
        console.print(Markdown("## -----DEBUG BLOCKS------"))
        console.print_json(data=format_slack_message(message))
        console.print(Markdown("## -----DEBUG BLOCKS------"))
    except:
        pass


class _ThreadInteractions(BaseModel):
    confirm: Any = None
    decline: Any = None
    cancel: Any = None
    text: Any = None


class SlackUIAgent(UIAgent):
    # these need to be populated after the agent is created by the factory
    app: AsyncApp = None
    context: "SlackContext" = None
    _handlers: _ThreadInteractions = PrivateAttr(default_factory=_ThreadInteractions)
    _input_callback: Any = PrivateAttr(default=None)
    _current_input_message: Any = PrivateAttr(default=None)
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    async def _send_to_user(self, message: GroupchatMessageTypes) -> None:
        try:
            if isinstance(message, AgentTrace):
                formatted_blocks = format_slack_message(message)
                await self.send_to_thread(**formatted_blocks)
            else:
                logger.warning(
                    f"Message type {type(message)} not supported for sending to user.")
        except Exception as e:  # noqa
            _fn_debug_blocks(message)

    async def send_to_thread(self, text=None, blocks=None, **kwargs):
        return await post_message_with_retry(
            app=self.app,
            context=self.context,
            text=text,
            blocks=blocks,
            **kwargs,
        )

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> None:
        """Send output to the Slack thread"""
        if isinstance(message, AgentTrace | AgentInput):
            await self._send_to_user(message)

    async def _request_input(
        self,
        message: ManagerRequest,
        **kwargs,
    ) -> None:
        """Ask for user input from the UI."""
        if isinstance(message, ManagerRequest):
            extra_blocks = []
            if message.prompt:
                extra_blocks = dict_to_blocks(message.prompt)
            if isinstance(message, ManagerRequest) and message.options is not None:
                if isinstance(message.options, bool):
                    # If there are binary options, display buttons
                    confirm_blocks = confirm_bool(
                        message=message.content,
                    )
                elif isinstance(message.options, list):
                    # If there are multiple options, display a dropdown
                    confirm_blocks = confirm_options(
                        message=message.content,
                        options=message.options,
                    )
                else:
                    raise ValueError("Invalid options type")
            else:
                # Assume binary yes / no
                confirm_blocks = confirm_bool(
                    message=message.content,
                )
                await self._cancel_input_request()

                # try:
                #     # we need to update the current message instead of
                #     # opening a new one.
                #     fn = self.app.client.chat_update(
                #         channel=self.context.channel_id,
                #         ts=self._current_input_message.data["ts"],
                #         text=confirm_blocks["text"],
                #         blocks=confirm_blocks["blocks"],
                #     )
                #     await request_with_retry(fn)
                # except:
                #     pass
            # We don't have an open input message. Send a new one.
            self._current_input_message = await self.send_to_thread(
                text=confirm_blocks["text"],
                blocks=confirm_blocks["blocks"],
            )

    async def _cancel_input_request(self):
        if self._current_input_message is not None:
            await self.app.client.chat_delete(
                channel=self.context.channel_id,
                ts=self._current_input_message.data["ts"],
            )
            self._current_input_message = None

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken = None, **kwargs) -> AgentTrace | None:
        """Tell the user we're expecting some data, but don't wait around"""
        if isinstance(message, ManagerRequest):
            await self._request_input(message)

    async def initialize(self, session_id: str, input_callback, **kwargs) -> None:
        """Initialize the interface and register handlers"""
        self._input_callback = input_callback

        _active_thread_registry[self.context.thread_ts] = self

        # Register this agent's thread for message handling
        self.register_chat_thread_handler(self.context.thread_ts)

    async def cleanup(self) -> None:
        """Clean up resources and unregister handlers"""
        # Remove this thread from the active registry
        if self.context.thread_ts in _active_thread_registry:
            del _active_thread_registry[self.context.thread_ts]

    def register_chat_thread_handler(self, thread_ts):
        """Connect messages in a Slack thread to the agent's callback"""
        logger.debug(f"Registering thread handler for {thread_ts}")

        async def matcher(message):
            return (
                # It's a message in our thread, not from the bot.
                message.get("thread_ts") == thread_ts and message.get("subtype") != "bot_message"
            )

        async def feed_in(message, say):
            await self._cancel_input_request()
            await self._input_callback(ManagerMessage(confirm=False, params=message["text"]))

        # Button action handlers
        async def handle_decline(ack, body, client):
            await ack()

            if not body.get("message", {}).get("thread_ts") == thread_ts:
                # why are we here?
                logger.debug("Received message not for us.")
                return

            # Only proceed if we are sure this message has not been dealt with yet.
            if self._current_input_message is None:
                return

            # Clear reference to input message
            self._current_input_message = None

            # Update UI to show cancellation
            await client.chat_update(
                channel=self.context.channel_id,
                ts=body["message"]["ts"],
                text=body["message"]["text"] + "\nYou selected: No",
                blocks=body["message"]["blocks"][:-1]
                + [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": ":x: You selected: *No*"},
                    },
                ],
            )
            # Call callback with boolean False
            await self._input_callback(ManagerMessage(confirm=False))

        async def handle_confirm(ack, body, client):
            await ack()

            if not body.get("message", {}).get("thread_ts") == thread_ts:
                # why are we here?
                logger.debug("Received message not for us.")
                return

            # Only proceed if we are sure this message has not been dealt with yet.
            if self._current_input_message is None:
                return

            # Clear reference to input message
            self._current_input_message = None

            # Update UI to show confirmation
            await client.chat_update(
                channel=self.context.channel_id,
                ts=body["message"]["ts"],
                text=body["message"]["text"] + "\nYou selected: Yes",
                blocks=body["message"]["blocks"][:-1]
                + [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":white_check_mark: You selected: *Yes*",
                        },
                    },
                ],
                actions=None,
            )
            # Call callback with boolean True
            await self._input_callback(ManagerMessage(confirm=True))
            self._current_input_message = None

        async def handle_cancel(ack, body, client):
            await ack()
            if not body.get("message", {}).get("thread_ts") == thread_ts:
                # why are we here?
                logger.debug("Received message not for us.")
                return

            # Only proceed if we are sure this message has not been dealt with yet.
            if self._current_input_message is None:
                return

            # Clear reference to input message
            self._current_input_message = None

            # Update UI to show cancellation
            await client.chat_update(
                channel=self.context.channel_id,
                ts=body["message"]["ts"],
                text=body["message"]["text"] + "\nYou selected: Cancel",
                blocks=body["message"]["blocks"][:-1]
                + [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": ":x: You selected: *Cancel*"},
                    },
                ],
            )
            # Call callback with boolean Halt signal.
            await self._input_callback(ManagerMessage(confirm=False, halt=True))

        self._handlers.text = self.app.message(matchers=[matcher])(feed_in)
        self._handlers.confirm = self.app.action("confirm_action")(handle_confirm)
        self._handlers.decline = self.app.action("decline_action")(handle_decline)
        self._handlers.cancel = self.app.action("cancel_action")(handle_cancel)

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        source: str = "unknown",
        **kwargs,
    ) -> OOBMessages:
        """Handle non-standard messages if needed (e.g., from orchestrator)."""
        # Ask for input if we need to
        if isinstance(message, ManagerRequest):
            await self._request_input(message)
        else:
            # otherwise just send to UI
            await self._send_to_user(message)

        return None


def reregister_all_active_threads():
    """Re-register handlers for all active threads after reconnection"""
    logger.info(
        f"Re-registering handlers for {len(_active_thread_registry)} active threads",
    )
    for thread_ts, agent in list(_active_thread_registry.items()):
        try:
            agent.register_chat_thread_handler(thread_ts)
        except Exception as e:
            logger.error(
                f"Failed to re-register handlers for thread {thread_ts}: {e!s}",
            )
