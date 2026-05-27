"""SpyAgent — passively listens and saves ExecutionTrace messages.

Phase 3: Removed direct RoutedAgent inheritance. SpyAgent is now a
standalone class with a handle_message() method. For compatibility
with the autogen GroupChat orchestrator, it retains a register()
classmethod that delegates to RoutedAgent.register().
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Phase 4 removal target: autogen compatibility for register()
from autogen_core import RoutedAgent

from buttermilk._core.runtime_types import MessageContext, message_handler
from buttermilk._core.storage_config import StorageConfig, StorageFactory

if TYPE_CHECKING:
    from autogen_core import AgentRuntime

from buttermilk import (
    bm,
    logger,
)
from buttermilk._core.agent import ProcessingError
from buttermilk._core.contract import (
    ErrorEvent,
    ExecutionTrace,
)
from buttermilk.utils.uploader import (
    AsyncDataUploader,
)

BATCH_SIZE = 10


class SpyAgent(RoutedAgent):
    """Passively captures ExecutionTrace messages and persists them.

    Inherits RoutedAgent as a compatibility adapter for the autogen
    GroupChat orchestrator (Phase 4 removal target).
    """

    def __init__(
        self,
        save: StorageConfig,
        **_kwargs: Any,
    ) -> None:
        super().__init__(description="Save results to storage")

        save = StorageFactory.create_config(save)
        self.storage = bm.get_storage(save)
        self.manager = AsyncDataUploader(storage=self.storage, buffer_size=BATCH_SIZE)

    @classmethod
    async def register(
        cls,
        runtime: "AgentRuntime",
        type: str,
        factory: Callable[[], Any],
        skip_class_subscriptions: bool = False,
        skip_direct_message_subscription: bool = False,
    ) -> Any:
        """Register SpyAgent with the autogen runtime (Phase 4 removal target)."""
        return await RoutedAgent.register(
            runtime=runtime,
            type=type,
            factory=factory,
            skip_class_subscriptions=skip_class_subscriptions,
            skip_direct_message_subscription=skip_direct_message_subscription,
        )

    @message_handler
    async def agent_output_handler(
        self, message: ExecutionTrace, ctx: MessageContext
    ) -> ErrorEvent | None:
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
