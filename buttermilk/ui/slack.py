import asyncio
from typing import Any

from pydantic import BaseModel
from slack_bolt.async_app import AsyncApp

from buttermilk.ui.chatio import GroupChatMessage, IOInterface


class SlackContext(BaseModel):
    channel_id: str
    thread_ts: str
    user_id: str | None = None


class SlackInterface(IOInterface):
    def __init__(self, app: AsyncApp, context: SlackContext):
        self.app = app
        self.context = context
        self.input_queue = asyncio.Queue()
        self.max_message_length = 3000

    async def get_input(self, prompt: str = "") -> GroupChatMessage:
        # If prompt is provided, send it first
        if prompt:
            await self.send_output(GroupChatMessage(content=prompt, step="User"))

        # Wait for a message to be added to the queue
        message = await self.input_queue.get()
        return message

    async def send_output(self, message: GroupChatMessage, source: str = "") -> None:
        content = message.content

        # Add source prefix if provided
        if source:
            content = f"*{source}*:\n{content}"

        # Split long messages
        messages = self._split_message(content)

        for msg in messages:
            await self.app.client.chat_postGroupChatMessage(
                channel=self.context.channel_id,
                thread_ts=self.context.thread_ts,
                text=msg,
            )

    def _split_message(self, text: str) -> list[str]:
        """Split message into chunks if it's too long for Slack"""
        if len(text) <= self.max_message_length:
            return [text]

        chunks = []
        while text:
            # Try to find a good break point
            if len(text) <= self.max_message_length:
                chunks.append(text)
                break

            # Find last newline or space in the allowed range
            pos = text[: self.max_message_length].rfind("\n")
            if pos < 0:
                pos = text[: self.max_message_length].rfind(" ")
            if pos < 0:
                pos = self.max_message_length

            chunks.append(text[:pos])
            text = text[pos:].lstrip()

        return chunks

    async def initialize(self) -> None:
        # Send initial message if needed
        pass

    async def cleanup(self) -> None:
        # Clean up resources if needed
        pass

    def add_message(self, text: str, metadata: dict[str, Any] = None) -> None:
        """Add a message to the queue from Slack events"""
        self.input_queue.put_nowait(
            GroupChatMessage(
                content=text,
                metadata=metadata or {},
            )
        )
