"""Native runtime types for buttermilk agent infrastructure.

Phase 3 of autogen removal: these types replace direct autogen_core imports
across agent subclasses. During the transition, some are re-exported from
autogen_core for runtime compatibility with the GroupChat orchestrator.
Phase 4 will replace these with fully native implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Re-exports from autogen_core — Phase 4 removal targets
# The orchestrator (groupchat.py) still uses the autogen runtime, which
# passes autogen MessageContext objects to @message_handler methods.
# Agent subclasses import these from here instead of autogen_core directly.
# ---------------------------------------------------------------------------
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    TopicId,
    message_handler,
)

# ---------------------------------------------------------------------------
# Native types — these are buttermilk-owned and will persist after Phase 4
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = "default"


@dataclass
class AgentIdentity:
    """Lightweight agent identity replacing autogen AgentId/AgentType/AgentMetadata."""

    key: str
    type: str = ""
    description: str = ""


class ChatHistory:
    """Simple chat history replacing autogen's UnboundedChatCompletionContext.

    Stores LLMMessage objects in order with add/get interface matching the
    autogen ChatCompletionContext protocol.
    """

    def __init__(self) -> None:
        self._messages: list[Any] = []

    async def add_message(self, message: Any) -> None:
        self._messages.append(message)

    async def get_messages(self) -> list[Any]:
        return list(self._messages)

    async def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


__all__ = [
    "AgentIdentity",
    "ChatHistory",
    "DEFAULT_TOPIC",
    "DefaultTopicId",
    "MessageContext",
    "TopicId",
    "message_handler",
]
