"""Defines the SpyAgent, an agent designed to passively listen and save data.

The `SpyAgent` operates within a multi-agent system (specifically compatible with
Autogen's `RoutedAgent` structure). Its primary function is to "lurk" in a
group chat or message bus, capture `AgentTrace` messages produced by other agents,
and persist them using an asynchronous data uploader.
"""

from typing import Any  # For type hinting

from autogen_core import (  # Autogen core components
    MessageContext,
    RoutedAgent,
    message_handler,  # Decorator to register methods as message handlers.
)

from buttermilk._core import logger  # Buttermilk's centralized logger
from buttermilk._core.agent import ProcessingError  # Buttermilk custom exception
from buttermilk._core.contract import AgentTrace, ErrorEvent  # Buttermilk message contracts
from buttermilk.utils.uploader import AsyncDataUploader  # Utility for asynchronous data upload

BATCH_SIZE = 10
"""Default buffer size for the `AsyncDataUploader` before flushing data."""


class SpyAgent(RoutedAgent):
    """An agent that passively listens to a message bus (e.g., group chat)
    and saves `AgentTrace` messages to a configured destination.

    The `SpyAgent` does not typically produce messages itself but acts as a data
    collector or logger for the activities of other agents. It uses an
    `AsyncDataUploader` to buffer messages and upload them in batches.

    The destination for saving data (e.g., BigQuery table, local file system,
    cloud storage) is defined by the `save_dest` parameter during initialization.

    Attributes:
        manager (AsyncDataUploader): An instance of `AsyncDataUploader` used to
            handle the asynchronous saving of captured messages.
        description (str): A description for the agent, set during initialization.
                           Defaults to "Save results to BQ" in the original code,
                           but could be made more generic.

    """

    def __init__(
        self,
        flow_name: str = "",
        **_kwargs: Any,
    ) -> None:
        """Initializes the SpyAgent.

        Args:
            flow_name: Name of the flow this agent is part of, provided by orchestrator
            **_kwargs: Additional keyword arguments passed to the `RoutedAgent`
                superclass constructor.

        """
        super().__init__(description="Save results to storage")
        self.flow_name = flow_name
        self.manager = AsyncDataUploader(buffer_size=BATCH_SIZE)
        
        # Configure storage based on flow_name provided by orchestrator
        if self.flow_name:
            self.manager.configure_storage(self.flow_name)
            logger.debug(f"SpyAgent: Configured storage for flow '{self.flow_name}'")
        else:
            logger.debug("SpyAgent: No flow_name provided, using session-level storage fallback")

    @message_handler  # Autogen decorator to register this method as a handler
    async def agent_output_handler(self, message: AgentTrace, ctx: MessageContext) -> ErrorEvent | None:  # Changed to Any to handle type check first
        """Message handler that captures `AgentTrace` messages and saves them.

        This method is decorated with `@message_handler`, making it the entry point
        for `AgentTrace` messages routed to this agent within an Autogen system.

        It performs the following actions:
        1.  Checks if the incoming `message` is an instance of `AgentTrace`.
        2.  If it is an `AgentTrace` and has `outputs` (i.e., it's not an empty trace),
            it performs a data cleaning step: if any `Record` objects within
            `message.inputs.records` have both "text" and "content" attributes,
            the "text" attribute is excluded before saving (this addresses a
            potential data conflict or redundancy).
        3.  The (potentially modified) `AgentTrace` message is then added to the
            `self.manager` (AsyncDataUploader) for asynchronous saving.
        4.  If the message has no `outputs`, it's logged and ignored.
        5.  If the message is not an `AgentTrace`, an error is logged, an `ErrorEvent`
            is published back to the topic from which the message came (if `publish_message`
            is available), and a `ProcessingError` is raised.

        Args:
            message (AgentTrace): The incoming message object. This handler specifically
                looks for `AgentTrace` instances.
            ctx (MessageContext): The context associated with the message, providing
                information like the topic ID and a method to publish messages.

        Returns:
            ErrorEvent | None: An `ErrorEvent` if an incompatible message type is
            received (and `publish_message` is available). Returns `None` otherwise,
            as this agent's primary role is to save data, not to produce direct
            reply messages in the main flow.
        
        Raises:
            ProcessingError: If an incompatible message type (not `AgentTrace`)
                is received.

        """
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
