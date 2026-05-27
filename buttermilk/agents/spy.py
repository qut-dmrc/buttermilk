"""SpyAgent — passively listens and saves ExecutionTrace messages."""

from collections.abc import Awaitable, Callable
from typing import Any

from buttermilk import (
    bm,
    logger,
)
from buttermilk._core.agent import ProcessingError
from buttermilk._core.contract import (
    ErrorEvent,
    ExecutionTrace,
)
from buttermilk._core.runtime_types import (
    AgentIdentity,
    MessageContext,
    _build_handler_registry,
    message_handler,
)
from buttermilk._core.storage_config import StorageConfig, StorageFactory
from buttermilk.utils.uploader import (
    AsyncDataUploader,
)

BATCH_SIZE = 10


class SpyAgent:
    """Passively captures ExecutionTrace messages and persists them."""

    def __init__(
        self,
        save: StorageConfig,
        publish_fn: Callable[..., Awaitable[None]] | None = None,
        **_kwargs: Any,
    ) -> None:
        save = StorageFactory.create_config(save)
        self.storage = bm.get_storage(save)
        self.manager = AsyncDataUploader(storage=self.storage, buffer_size=BATCH_SIZE)
        self._publish_fn = publish_fn
        self._handler_registry: dict[type, str] | None = None

    @property
    def identity(self) -> AgentIdentity:
        return AgentIdentity(key="spy", type="SpyAgent", description="Save results to storage")

    def _get_handler_registry(self) -> dict[type, str]:
        if self._handler_registry is None:
            self._handler_registry = _build_handler_registry(type(self))
        return self._handler_registry

    async def dispatch(self, message: Any, ctx: MessageContext) -> Any:
        """Dispatch an incoming message to the appropriate handler."""
        registry = self._get_handler_registry()
        msg_type = type(message)
        method_name = registry.get(msg_type)
        if method_name is None:
            for handled_type, name in registry.items():
                if issubclass(msg_type, handled_type):
                    method_name = name
                    break
        if method_name is not None:
            method = getattr(self, method_name)
            match_pred = getattr(method, "_match_predicate", None)
            if match_pred and not match_pred(message, ctx):
                return None
            return await method(message, ctx)
        return None

    async def close(self) -> None:
        """Flush pending writes."""
        await self.manager.flush()

    @message_handler
    async def agent_output_handler(self, message: ExecutionTrace, ctx: MessageContext) -> ErrorEvent | None:
        """Capture ExecutionTrace messages and save them."""
        if isinstance(message, ExecutionTrace):
            if message.outputs:
                logger.debug(f"SpyAgent received message of type: {type(message)} on topic {ctx.topic_id}")
                await self.manager.add(message)
            else:
                logger.debug(f"SpyAgent received message with no outputs: {message} on topic {ctx.topic_id}")
        else:
            msg = f"Spy database save agent received incompatible output type: {type(message)} on topic {ctx.topic_id}"
            raise ProcessingError(msg)
        return None
