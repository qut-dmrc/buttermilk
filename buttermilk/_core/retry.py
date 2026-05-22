import asyncio
import socket
from collections.abc import Callable
from logging import DEBUG
from typing import Any

import aiohttp
import requests  # type: ignore[import-untyped]
import urllib3
from anthropic._exceptions import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic._exceptions import (
    InternalServerError as AnthropicInternalServerError,
)
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,
)
from anthropic._exceptions import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic._exceptions import (
    ServiceUnavailableError as AnthropicServiceUnavailableError,
)
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from google.auth.exceptions import TransportError as GoogleAuthTransportError
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from pydantic import BaseModel

try:
    from pyzotero import zotero_errors

    PYZOTERO_AVAILABLE = True
except ImportError:
    zotero_errors = None
    PYZOTERO_AVAILABLE = False

from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from buttermilk._core.exceptions import RateLimit
from buttermilk._core.log import logger


class RetryWrapper(BaseModel):
    """Wraps a client and adds robust retry logic for handling API failures.

    Args:
        client: The object to wrap
        cooldown_seconds: Time to wait between calls
        max_retries: Maximum number of retry attempts for failed API calls
        min_wait_seconds: Minimum wait time between retries (exponential backoff)
        max_wait_seconds: Maximum wait time between retries
        jitter_seconds: Random jitter added to wait times to prevent thundering herd

    """

    client: Any
    cooldown_seconds: float = 0.5
    max_retries: int = 3
    min_wait_seconds: float = 5.0
    max_wait_seconds: float = 60.0
    jitter_seconds: float = 5.0

    model_config = {"arbitrary_types_allowed": True}

    def _get_retry_config(self) -> dict:
        """Get the retry configuration for tenacity."""
        # Base exception types that are always available
        retry_exceptions = [
            RateLimit,
            TimeoutError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            urllib3.exceptions.ProtocolError,
            urllib3.exceptions.TimeoutError,
            urllib3.exceptions.NameResolutionError,
            urllib3.exceptions.NewConnectionError,
            OpenAIAPIConnectionError,
            OpenAIRateLimitError,
            AnthropicAPIConnectionError,
            AnthropicRateLimitError,
            AnthropicOverloadedError,
            AnthropicInternalServerError,
            AnthropicServiceUnavailableError,
            TooManyRequests,
            ResourceExhausted,
            GoogleAuthTransportError,
            ConnectionResetError,
            ConnectionError,
            ConnectionAbortedError,
            socket.gaierror,
            aiohttp.ClientError,
        ]

        # Add pyzotero exceptions if available
        if PYZOTERO_AVAILABLE:
            retry_exceptions.extend(
                [
                    zotero_errors.HTTPError,
                    zotero_errors.TooManyRequestsError,
                ]
            )

        return {
            "retry": retry_if_exception_type(tuple(retry_exceptions)),
            "stop": stop_after_attempt(self.max_retries),
            "wait": wait_exponential_jitter(
                initial=self.min_wait_seconds,
                max=self.max_wait_seconds,
                jitter=self.jitter_seconds,
            ),
            "before_sleep": before_sleep_log(logger, DEBUG, exc_info=False),
            "reraise": True,
        }

    async def _execute_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with retry logic."""
        try:
            async for attempt in AsyncRetrying(**self._get_retry_config()):
                with attempt:
                    # Execute the function
                    result = await func(*args, **kwargs)
                    # Add a small delay on success to prevent rate limiting
                    await asyncio.sleep(self.cooldown_seconds)
                    return result
        except RetryError as e:
            logger.error(f"All retry attempts failed: {e!s}")
            # Re-raise the last exception
            raise e.last_attempt.exception()

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes to the wrapped client only if they don't exist on self.
        This prevents accidentally redirecting access to important instance attributes.
        """
        # Check if this is an attribute that should exist on self
        if (
            name in self.__dict__
            or name in self.__annotations__
            or name in self.__class__.__dict__
            or (name.startswith("_") and hasattr(type(self), name))
            or (hasattr(type(self), "__private_attributes__") and name in type(self).__private_attributes__)
        ):
            # Let the normal attribute lookup process handle this (which will raise
            # AttributeError if appropriate)
            return self.__getattribute__(name)

        # Otherwise delegate to the client
        return getattr(self.client, name)
