"""Native runtime types for buttermilk agent infrastructure."""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Topic types — string-based with .type property for backward compatibility
# ---------------------------------------------------------------------------


class TopicId(str):
    """Topic identifier — a str subclass with .type for backward compat."""

    def __new__(cls, type: str = "default"):  # noqa: A002
        return super().__new__(cls, type)

    @property
    def type(self) -> str:
        return str(self)


DefaultTopicId = TopicId

# ---------------------------------------------------------------------------
# MessageContext — passed to @message_handler methods
# ---------------------------------------------------------------------------


@dataclass
class MessageContext:
    """Context passed to message handlers by the orchestrator."""

    topic_id: str = ""
    sender: AgentIdentity | None = None
    cancellation_token: Any = None


# ---------------------------------------------------------------------------
# message_handler decorator
# ---------------------------------------------------------------------------


def message_handler(func=None, *, match=None):
    """Mark a method as a message handler.

    The orchestrator dispatches messages to handlers by matching the
    incoming message type against the 'message' parameter's type hint.
    Optional `match` predicate filters messages before dispatch.
    """
    def decorator(fn):
        fn._is_message_handler = True
        fn._match_predicate = match
        return fn

    if func is not None:
        return decorator(func)
    return decorator


def _build_handler_registry(cls: type) -> dict[type, str]:
    """Build a message-type → method-name dispatch table for a class.

    Walks the MRO so that subclass handlers take precedence over parent
    handlers for the same message type.
    """
    registry: dict[type, str] = {}

    # Walk MRO in reverse so child classes override parent registrations
    for klass in reversed(cls.__mro__):
        for name, method in vars(klass).items():
            if not getattr(method, "_is_message_handler", False):
                continue
            try:
                hints = typing.get_type_hints(method)
            except Exception:
                continue
            msg_type = hints.get("message")
            if msg_type is None:
                continue
            # Handle Union types (X | Y or Union[X, Y])
            origin = getattr(msg_type, "__origin__", None)
            if origin is typing.Union:
                for t in msg_type.__args__:
                    if isinstance(t, type):
                        registry[t] = name
            elif isinstance(msg_type, type):
                registry[msg_type] = name
    return registry


# ---------------------------------------------------------------------------
# Native types
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = "default"


@dataclass
class AgentIdentity:
    """Lightweight agent identity."""

    key: str
    type: str = ""
    description: str = ""


class ChatHistory:
    """Simple chat history store.

    Stores LLMMessage objects in order with add/get interface.
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
