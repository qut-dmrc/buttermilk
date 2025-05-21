from autogen_core import (
    MessageContext,
    RoutedAgent,
    message_handler,  # Decorator to register methods as message handlers.
)

from buttermilk._core import logger
from buttermilk._core.agent import ProcessingError
from buttermilk._core.config import SaveInfo
from buttermilk._core.contract import AgentTrace, ErrorEvent
from buttermilk.utils.uploader import AsyncDataUploader

BATCH_SIZE = 10


class SpyAgent(RoutedAgent):
    """Agent just lurks in the group chat, saving things to a database."""

    def __init__(
        self,
        save_dest: SaveInfo, **kwargs,
    ) -> None:
        super().__init__(description="Save results to BQ")
        self.manager = AsyncDataUploader(buffer_size=BATCH_SIZE, save_dest=save_dest)

    @message_handler
    async def agent_output(self, message: AgentTrace, ctx: MessageContext) -> ErrorEvent | None:
        """Captures outputs from other agents and saves them."""
        if isinstance(message, AgentTrace):
            if message.outputs:
                logger.debug(f"SpyAgent received message of type: {type(message)} on topic {ctx.topic_id}")  # Log received type and topic
                # Check if there's records in the inputs and then make sure they don't have both 'text' and 'content' fields.
                if message.inputs and message.inputs.records:
                    for record in message.inputs.records:
                        if hasattr(record, "text") and hasattr(record, "content"):
                            # This shouldn't happen because the pydantic model excludes text.
                            # But for some reason it does, so we need to handle it.
                            message.inputs.records = [x.model_dump(exclude="text") for x in message.inputs.records]
                            break
                await self.manager.add(message)
            else:
                logger.debug(f"SpyAgent received message with no outputs: {message} on topic {ctx.topic_id}")
        else:
            msg = f"Spy database save agent received incompatible output type: {type(message)} on topic {ctx.topic_id}"
            logger.error(msg)
            await self.publish_message(ErrorEvent(source=self.id.type, content=msg), topic_id=ctx.topic_id)
            raise ProcessingError(msg)
        return None
