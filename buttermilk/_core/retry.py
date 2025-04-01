import asyncio
import logging
from collections.abc import Callable
from typing import Any, Self

import requests
import urllib3
from anthropic._exceptions import (
    APIConnectionError as AnthropicAPIConnectionError,
    InternalServerError as AnthropicInternalServerError,
    OverloadedError as AnthropicOverloadedError,
    RateLimitError as AnthropicRateLimitError,
    ServiceUnavailableError as AnthropicServiceUnavailableError,
)
from google.api_core.exceptions import TooManyRequests
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    RateLimitError as OpenAIRateLimitError,
)
from pydantic import BaseModel, Field, model_validator
from replicate.exceptions import ModelError, ReplicateError
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from buttermilk._core.log import logger
from buttermilk.exceptions import RateLimit


class RetryWrapper(BaseModel):
    """Wraps a client and adds rate limiting via a semaphore
    plus robust retry logic for handling API failures.

    Args:
        client: The object to wrap
        max_concurrent_calls: Maximum number of concurrent calls allowed
        cooldown_seconds: Time to wait between calls
        max_retries: Maximum number of retry attempts for failed API calls
        min_wait_seconds: Minimum wait time between retries (exponential backoff)
        max_wait_seconds: Maximum wait time between retries
        jitter_seconds: Random jitter added to wait times to prevent thundering herd

    """

    client: Any
    max_concurrent_calls: int = 3
    cooldown_seconds: float = 0.1
    max_retries: int = 6
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 30.0
    jitter_seconds: float = 1.0

    semaphore: asyncio.Semaphore = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def create_semaphore(self) -> Self:
        # Use the proper setter for private attributes
        self.semaphore = asyncio.Semaphore(
            self.max_concurrent_calls,
        )
        return self

    def _get_retry_config(self) -> dict:
        """Get the retry configuration for tenacity."""
        return {
            "retry": retry_if_exception_type(
                (
                    RateLimit,
                    TimeoutError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    urllib3.exceptions.ProtocolError,
                    urllib3.exceptions.TimeoutError,
                    OpenAIAPIConnectionError,
                    OpenAIRateLimitError,
                    AnthropicAPIConnectionError,
                    AnthropicRateLimitError,
                    AnthropicOverloadedError,
                    AnthropicInternalServerError,
                    AnthropicServiceUnavailableError,
                    ModelError,
                    ReplicateError,
                    TooManyRequests,
                    ConnectionResetError,
                    ConnectionError,
                    ConnectionAbortedError,
                ),
            ),
            "stop": stop_after_attempt(self.max_retries),
            "wait": wait_exponential_jitter(
                initial=self.min_wait_seconds,
                max=self.max_wait_seconds,
                jitter=self.jitter_seconds,
            ),
            "before_sleep": before_sleep_log(logger, logging.WARNING),
            "reraise": True,
        }

    async def _execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a function with retry logic."""
        try:
            async for attempt in AsyncRetrying(**self._get_retry_config()):
                with attempt:
                    async with self.semaphore:
                        # Execute the function
                        result = await func(*args, **kwargs)
                        # Add a small delay on success to prevent rate limiting
                        await asyncio.sleep(self.cooldown_seconds)
                        return result
        except RetryError as e:
            logger.error(f"All retry attempts failed: {e!s}")
            # Re-raise the last exception
            raise e.last_attempt.exception()
        except Exception as e:
            logger.error(
                f"{type(self.client).__qualname__} client hit unexpected exception: {e!s}",
            )
            raise e

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
        ):
            # Let the normal attribute lookup process handle this (which will raise
            # AttributeError if appropriate)
            return self.__getattribute__(name)

        # Otherwise delegate to the client
        return getattr(self.client, name)
