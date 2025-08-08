import weave  # For tracing - core dependency
from weave.trace.weave_client import Call, WeaveClient, WeaveObject

# Buttermilk core imports
from buttermilk._core.contract import (
    AgentInput,
)
from buttermilk._core.dmrc import get_bm
from buttermilk._core.log import logger  # Buttermilk logger instance
from buttermilk._core.retry import RetryWrapper


def get_weave() -> WeaveClient:
    """Provides access to the Weights & Biases Weave client for tracing.

    Initializes Weave with a collection name derived from `self.name` (flow name)
    and `self.job` (job name). Sets up credentials from environment variables or
    secret manager before initialization to avoid interactive login flows.
    Handles connection failures gracefully by falling back to a mock client.

    Returns:
        Any: The initialized Weave client instance, or a mock client if initialization fails.

    """
    bm = get_bm()
    collection_name = f"{bm.run_info.name}-{bm.run_info.job}"  # Construct collection name

    # client = weave.init(collection_name)
    # We disable weave autopatching for Autogen because it's too noisy and slow
    # We will instead trace manually.
    client = weave.init(collection_name, autopatch_settings={"autogen": {"enabled": False}})
    logger.debug("Weave initialized successfully")
    return client


async def get_parent_call_weave(
    message: AgentInput | None = None,
) -> Call | WeaveObject:

    if message and message.parent_call_id:

        async def get_weave_call_with_retry(call_id: str) -> Call | WeaveObject:
            """Retry getting weave call to handle async upload timing."""
            return get_bm().weave.get_call(call_id)

        # Use RetryWrapper with shorter delays for weave call retrieval
        retry_wrapper = RetryWrapper(
            client=None,  # Not using client, just the retry logic
            max_retries=3,
            min_wait_seconds=0.1,
            max_wait_seconds=1.0,
            jitter_seconds=0.1,
            cooldown_seconds=0,
        )

        try:
            parent_call = await retry_wrapper._execute_with_retry(
                get_weave_call_with_retry,
                message.parent_call_id,
            )
        except Exception as e:  # Broad exception for Weave call retrieval
            logger.error(f"Could not retrieve parent call ID {message.parent_call_id} after retries. Error: {e}")
            parent_call = weave.get_current_call()  # Fallback to current call if specified parent not found
    else:
        parent_call = weave.get_current_call()

    return parent_call
