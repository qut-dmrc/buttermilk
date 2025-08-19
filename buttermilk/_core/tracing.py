import weave  # For tracing - core dependency
from weave.trace.weave_client import Call, WeaveObject

from buttermilk import get_bm, logger

# Buttermilk core imports
from buttermilk._core.contract import (
    AgentInput,
)
from buttermilk._core.retry import RetryWrapper


async def get_parent_call_weave(
    message: AgentInput | None = None,
) -> Call | WeaveObject:
    if get_bm().get_weave_client() is None:
        logger.warning("Weave client is not initialized, cannot retrieve parent call.")
        return None
    current_call = weave.get_current_call()

    if message is None or message.parent_call_id is None:
        # If no message or no parent call ID, return current call as parent
        return current_call

    # Unless calls are out of order, there's a good chance the current call is the parent.
    if message.parent_call_id == current_call.id:
        return current_call

    # If not, we have to go out and find the parent call from the Weave API.
    # This sometimes fails because the call hasn't been uploaded yet.
    # We retry a few times to handle this.
    #
    # TODO: check whether this is actually necessary; can we just use the ID to
    # associate calls together in a meaningful hiearchy?

    async def get_weave_call_with_retry(call_id: str) -> Call | WeaveObject:
        """Retry getting weave call to handle async upload timing."""
        bm = get_bm()
        return bm.get_weave_client().get_call(call_id)

    # Use RetryWrapper with shorter delays for weave call retrieval
    retry_wrapper = RetryWrapper(
        client=None,  # Not using client, just the retry logic
        max_retries=3,
        min_wait_seconds=3.0,
        max_wait_seconds=10.0,
        jitter_seconds=0.1,
        cooldown_seconds=0,
    )

    try:
        parent_call = await retry_wrapper._execute_with_retry(
            get_weave_call_with_retry,
            message.parent_call_id,
        )
        return parent_call
    except Exception as e:  # Broad exception for Weave call retrieval
        logger.error(f"Could not retrieve parent call ID {message.parent_call_id} after retries. Error: {e}")
        return current_call
