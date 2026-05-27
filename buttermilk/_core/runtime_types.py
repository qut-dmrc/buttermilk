"""Native runtime types for buttermilk agent infrastructure."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
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
            # Handle Union types — both typing.Union[X, Y] and PEP 604 X | Y
            origin = getattr(msg_type, "__origin__", None)
            if origin is typing.Union or isinstance(msg_type, types.UnionType):
                for t in msg_type.__args__:
                    if isinstance(t, type):
                        registry[t] = name
            elif isinstance(msg_type, type):
                registry[msg_type] = name
    return registry


async def dispatch_message(target: Any, message: Any, ctx: MessageContext) -> Any:
    """Dispatch a message to the appropriate @message_handler on *target*.

    Shared implementation used by Agent.dispatch() and SpyAgent.dispatch()
    to avoid duplicating the handler-registry lookup logic.

    Args:
        target: The object whose handler registry and methods to use.
        message: The incoming message to dispatch.
        ctx: Message context (topic, sender, etc.).

    Returns:
        The return value of the matched handler, or None if no handler matched.
    """
    registry = target._get_handler_registry()
    msg_type = type(message)
    method_name = registry.get(msg_type)
    if method_name is None:
        for handled_type, name in registry.items():
            if issubclass(msg_type, handled_type):
                method_name = name
                break
    if method_name is not None:
        method = getattr(target, method_name)
        match_pred = getattr(method, "_match_predicate", None)
        if match_pred and not match_pred(message, ctx):
            return None
        return await method(message, ctx)
    return None


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
    "dispatch_message",
    "message_handler",
]
