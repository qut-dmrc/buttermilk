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
    dispatch_message,
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
        save: dict[str, Any],
        publish_fn: Callable[..., Awaitable[None]] | None = None,
        **_kwargs: Any,
    ) -> None:
        save_config = StorageFactory.create_config(save)
        self.storage = bm.get_storage(save_config)
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
        return await dispatch_message(self, message, ctx)

    async def close(self) -> None:
        """Flush pending writes."""
        await self.manager._flush()

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
