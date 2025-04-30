from autogen_core import (
    MessageContext,
    RoutedAgent,
    message_handler,  # Decorator to register methods as message handlers.
)

from buttermilk._core.agent import ProcessingError
from buttermilk._core.config import SaveInfo
from buttermilk._core.contract import AgentOutput, ErrorEvent
from buttermilk.bm import logger
from buttermilk.utils.uploader import AsyncDataUploader

BATCH_SIZE = 10


class SpyAgent(RoutedAgent):
    """Agent just lurks in the group chat, saving things to a database."""

    def __init__(
        self,
        save_dest: SaveInfo,
    ) -> None:
        super().__init__(description="Save results to BQ")
        self.manager = AsyncDataUploader(buffer_size=BATCH_SIZE, save_dest=save_dest)

    @message_handler
    async def agent_output(self, message: AgentOutput, ctx: MessageContext) -> ErrorEvent | None:
        logger.debug(f"SpyAgent received message of type: {type(message)} on topic {ctx.topic_id}")  # Log received type and topic

        """Captures outputs from other agents and saves them."""
        if isinstance(message, AgentOutput):
            await self.manager.add(message)
        else:
            msg = f"Spy database save agent received incompatible output type: {type(message)}"
            logger.error(msg)
            await self.publish_message(ErrorEvent(source=self.id.type, content=msg), topic_id=ctx.topic_id)
            raise ProcessingError(msg)
        return None
