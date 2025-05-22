"""Provides a Pydantic BaseModel wrapper for adding robust retry logic to client interactions.

This module defines `RetryWrapper`, a class that can wrap any client object
(e.g., an API client for an LLM or a cloud service) and automatically apply
retry mechanisms when specific exceptions occur during method calls. It uses
the `tenacity` library to implement exponential backoff with jitter.
"""

import asyncio
import logging
from collections.abc import Callable  # For typing callables
from typing import Any

import requests  # Common HTTP library
import urllib3  # Underlying HTTP library used by requests
from anthropic._exceptions import (  # Specific exceptions from Anthropic client
    APIConnectionError as AnthropicAPIConnectionError,
    InternalServerError as AnthropicInternalServerError,
    OverloadedError as AnthropicOverloadedError,
    RateLimitError as AnthropicRateLimitError,
    ServiceUnavailableError as AnthropicServiceUnavailableError,
)
from google.api_core.exceptions import TooManyRequests  # Google API core exceptions
from openai import (  # Specific exceptions from OpenAI client
    APIConnectionError as OpenAIAPIConnectionError,
    RateLimitError as OpenAIRateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field  # Pydantic components
from tenacity import (  # Retry library components
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from buttermilk._core.exceptions import RateLimit  # Custom Buttermilk RateLimit exception
from buttermilk._core.log import logger  # Centralized logger


class RetryWrapper(BaseModel):
    """Wraps a client object to add robust retry logic for its method calls.

    This class uses the `tenacity` library to automatically retry function calls
    when specific, transient exceptions occur (e.g., rate limit errors, connection
    errors, timeouts). It implements an exponential backoff strategy with jitter
    to avoid overwhelming services during retries.

    Attributes:
        client (Any): The client object whose methods will be wrapped with retry logic.
        cooldown_seconds (float): Time in seconds to wait after a successful call
            before allowing another call. This helps prevent immediate rate limiting
            on some services. Defaults to 0.5.
        max_retries (int): The maximum number of retry attempts for a failed API call
            before giving up. Defaults to 3.
        min_wait_seconds (float): The minimum (initial) wait time in seconds between
            retries when using exponential backoff. Defaults to 5.0.
        max_wait_seconds (float): The maximum wait time in seconds between retries,
            capping the exponential backoff. Defaults to 60.0.
        jitter_seconds (float): Random jitter in seconds added to wait times to
            prevent thundering herd problems (multiple clients retrying simultaneously).
            Defaults to 5.0.
        model_config (ConfigDict): Pydantic model configuration.
            - `arbitrary_types_allowed`: True - Allows the `client` attribute to be of any type.

    """

    client: Any = Field(description="The client object to which retry logic will be applied.")
    cooldown_seconds: float = Field(
        default=0.5,
        description="Time in seconds to wait after a successful call to prevent immediate rate limiting.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed API calls.",
    )
    min_wait_seconds: float = Field(
        default=5.0,
        description="Minimum wait time in seconds between retries (for exponential backoff).",
    )
    max_wait_seconds: float = Field(
        default=60.0,
        description="Maximum wait time in seconds between retries, capping exponential backoff.",
    )
    jitter_seconds: float = Field(
        default=5.0,
        description="Random jitter in seconds added to wait times to prevent thundering herd issues.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_retry_config(self) -> dict[str, Any]:
        """Constructs the configuration dictionary for `tenacity.AsyncRetrying`.

        This configuration specifies which exceptions should trigger a retry,
        the stopping condition (max number of attempts), the waiting strategy
        (exponential backoff with jitter), and logging behavior before sleep.

        Returns:
            dict[str, Any]: A dictionary of keyword arguments suitable for
            `tenacity.AsyncRetrying`.

        """
        return {
            "retry": retry_if_exception_type(
                (
                    # Custom Buttermilk exceptions
                    RateLimit,
                    # Standard Python exceptions
                    TimeoutError,
                    ConnectionResetError,
                    ConnectionError,
                    ConnectionAbortedError,
                    # `requests` library exceptions
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    # `urllib3` (used by requests) exceptions
                    urllib3.exceptions.ProtocolError,
                    urllib3.exceptions.TimeoutError,
                    # OpenAI client exceptions
                    OpenAIAPIConnectionError,
                    OpenAIRateLimitError,
                    # Anthropic client exceptions
                    AnthropicAPIConnectionError,
                    AnthropicRateLimitError,
                    AnthropicOverloadedError,
                    AnthropicInternalServerError,
                    AnthropicServiceUnavailableError,
                    # Google API core exceptions
                    TooManyRequests,
                    # Add other common transient exceptions here as needed
                    # e.g., specific cloud provider SDK exceptions for transient errors
                ),
            ),
            "stop": stop_after_attempt(self.max_retries),
            "wait": wait_exponential_jitter(
                initial=self.min_wait_seconds,  # Initial wait
                max=self.max_wait_seconds,     # Max wait between retries
                jitter=self.jitter_seconds,    # Random jitter
            ),
            "before_sleep": before_sleep_log(logger, logging.WARNING, exc_info=False),  # Log before retrying
            "reraise": True,  # Re-raise the last exception if all retries fail
        }

    async def _execute_with_retry(
        self,
        func: Callable[..., Any],  # The function to call
        *args: Any,               # Positional arguments for the function
        **kwargs: Any,            # Keyword arguments for the function
    ) -> Any:
        """Executes a given asynchronous function with the configured retry logic.

        It uses `tenacity.AsyncRetrying` to wrap the function call. If the call
        is successful, a brief cooldown period is observed before returning the result.
        If all retry attempts fail, the exception from the last attempt is re-raised.

        Args:
            func (Callable[..., Any]): The asynchronous function to execute.
            *args: Positional arguments to pass to `func`.
            **kwargs: Keyword arguments to pass to `func`.

        Returns:
            Any: The result of the successful execution of `func`.

        Raises:
            RetryError: If all retry attempts fail (though `reraise=True` in config
                means the actual last exception will be raised instead of RetryError itself).
            Exception: The last exception encountered if all retries are exhausted.

        """
        try:
            retry_config = self._get_retry_config()
            # Create an AsyncRetrying instance with the configuration
            async_retryer = AsyncRetrying(**retry_config)

            # Use the retryer to call the function
            # Note: tenacity's call method directly handles awaiting the async func
            return await async_retryer.call(self._call_func_with_cooldown, func, *args, **kwargs)

        except RetryError as e:  # This block might be less likely if reraise=True
            logger.error(f"All retry attempts for {func.__name__} failed: {e!s}")
            # If reraise is True (default), tenacity re-raises the last_attempt's exception directly.
            # So, this specific 'raise e.last_attempt.exception()' might be redundant if
            # tenacity already does this. However, explicit is often better than implicit.
            if e.last_attempt:  # Check if last_attempt exists
                raise e.last_attempt.exception()  # Re-raise the actual last exception
            raise  # Re-raise RetryError if last_attempt is not available for some reason

    async def _call_func_with_cooldown(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Helper to call the function and then apply cooldown."""
        result = await func(*args, **kwargs)
        await asyncio.sleep(self.cooldown_seconds)  # Cooldown after successful call
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute access to the wrapped `self.client` instance.

        If an attribute is not found on the `RetryWrapper` instance itself,
        this method attempts to retrieve it from the underlying `self.client`.
        This allows the `RetryWrapper` to be used as if it were the client
        object directly for most purposes (e.g., `wrapper.some_client_method()`).

        It includes checks to prevent delegation of attributes that are intended
        to be part of the `RetryWrapper` itself (e.g., Pydantic model fields,
        private attributes, methods of `RetryWrapper`).

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The attribute value from the wrapped client, or raises an
                 AttributeError if not found on either the wrapper or the client.
        
        Raises:
            AttributeError: If the attribute `name` is not found on the `RetryWrapper`
                            itself (for its own defined attributes/methods) or on the
                            underlying `client`.

        """
        # Check if the attribute is part of the RetryWrapper model itself or its annotations/methods
        # This prevents delegation of RetryWrapper's own attributes.
        if name in self.model_fields or \
           name in self.__annotations__ or \
           hasattr(self.__class__, name) or \
           (name.startswith("_") and hasattr(self, name)):  # Check for private attributes too
            # If it's an attribute of RetryWrapper, let Pydantic/Python handle it.
            # This will raise AttributeError if it truly doesn't exist on the wrapper.
            return super().__getattribute__(name)

        # If not an attribute of RetryWrapper, delegate to the wrapped client.
        # This will raise AttributeError if the client also doesn't have it.
        if hasattr(self, "client") and self.client is not None:
            return getattr(self.client, name)

        # Fallback if client is None or attribute is not found
        raise AttributeError(f"'{self.__class__.__name__}' object and its 'client' have no attribute '{name}'")
