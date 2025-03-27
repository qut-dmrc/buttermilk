import logging
from typing import Any

from pydantic import BaseModel

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from buttermilk.bm import logger


class SlackContext(BaseModel):
    channel_id: str
    thread_ts: str
    user_id: str | None = None
    event_ts: str | None = None
    say: Any = None



@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def post_message_with_retry(
    app,
    context: SlackContext,
    text,
    blocks=None,
    **kwargs,
):
    message = {
        "channel": context.channel_id,
        "text": text,
        "blocks": blocks,
        "thread_ts": context.thread_ts,
    }
    message.update(kwargs)
    return await app.client.chat_postMessage(**message)
