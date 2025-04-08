from typing import Any

import pydantic
from pydantic import BaseModel, PrivateAttr
from rich.console import Console
from rich.markdown import Markdown
from slack_bolt.async_app import AsyncApp

from autogen_core import CancellationToken, MessageContext
from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    UserInstructions,
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
    request_with_retry,
)

_active_thread_registry = {}


def _fn_debug_blocks(message: AgentOutput):
    return
    try:
        console = Console(highlight=True)
        console.print(Markdown("## -----DEBUG BLOCKS------"))
        console.print_json(data=format_slack_message(message.outputs))
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

    async def send_to_thread(self, text=None, blocks=None, **kwargs):
        return await post_message_with_retry(
            app=self.app,
            context=self.context,
            text=text,
            blocks=blocks,
            **kwargs,
        )

    async def listen(
        self,
        message: GroupchatMessageTypes,
        ctx: MessageContext = None,
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

    async def _request_input(
        self,
        message: AgentInput | ManagerRequest | ManagerMessage,
        **kwargs,
    ) -> None:
        """Ask for user input from the UI."""
        if isinstance(message, ManagerResponse):
            return
        elif isinstance(message, (AgentInput, ManagerRequest)):
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
        else:
            raise ValueError("Invalid message type")

    async def _cancel_input_request(self):
        if self._current_input_message is not None:
            fn = self.app.client.chat_delete(
                channel=self.context.channel_id,
                ts=self._current_input_message.data["ts"],
            )
            await request_with_retry(fn)
            self._current_input_message = None

    async def _process(
        self,
        input_data: AgentInput,
        **kwargs,
    ) -> AgentOutput | None:
        """Tell the user we're expecting some data, but don't wait around"""
        await self._request_input(input_data)
        yield # Required for async generator typing

    async def initialize(self, input_callback, **kwargs) -> None:
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
                message.get("thread_ts") == thread_ts
                and message.get("subtype") != "bot_message"
            )

        async def feed_in(message, say):
            await self._cancel_input_request()
            await self._input_callback(ManagerResponse(confirm=False, source="slack-thread", role=self.role))
            await self._input_callback(UserInstructions(content=message["text"], role=self.role, source="slack-thread"))

        # Button action handlers
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
            await self._input_callback(ManagerResponse(confirm=True, source="slack-thread", role=self.role))
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
            await self._input_callback(ManagerResponse(confirm=False, source="slack-thread", role=self.role))

        self._handlers.text = self.app.message(matchers=[matcher])(feed_in)
        self._handlers.confirm = self.app.action("confirm_action")(handle_confirm)
        self._handlers.decline = self.app.action("decline_action")(handle_confirm)
        self._handlers.cancel = self.app.action("cancel_action")(handle_cancel)


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
