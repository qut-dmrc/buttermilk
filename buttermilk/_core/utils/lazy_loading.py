"""Utilities for lazy loading and caching resources."""

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def cached_property(initialize_func: Callable[[Any], T]) -> property:
    """
    Decorator for creating a property with lazy initialization and caching.

    This creates a property that initializes its value on first access and
    caches it for subsequent accesses.

    Args:
        initialize_func: The function to call to initialize the value

    Returns:
        A property that lazily initializes and caches the value
    """
    attr_name = f"_{initialize_func.__name__}_cached"

    def getter(self: Any) -> T:
        # Get the cached value using getattr with default None
        cached_value = getattr(self, attr_name, None)
        if cached_value is None:
            # Initialize and cache
            cached_value = initialize_func(self)
            setattr(self, attr_name, cached_value)
        return cached_value

    # Use the original function's docstring
    getter.__doc__ = initialize_func.__doc__

    return property(getter)


def refreshable_cached_property(
    initialize_func: Callable[[Any], T],
    should_refresh: Callable[[Any, T], bool] | None = None,
) -> property:
    """
    Decorator for creating a property with lazy initialization, caching, and refresh capability.

    This builds on cached_property but adds the ability to refresh the cached value
    based on a condition function.

    Args:
        initialize_func: The function to call to initialize the value
        should_refresh: Optional function that takes the instance and current value
                        and returns True if the value should be refreshed

    Returns:
        A property that lazily initializes, caches, and refreshes the value as needed
    """
    attr_name = f"_{initialize_func.__name__}_cached"

    def getter(self: Any) -> T:
        # Get the cached value using getattr with default None
        cached_value = getattr(self, attr_name, None)

        # Initialize if not yet initialized or refresh if needed
        if cached_value is None or (should_refresh and should_refresh(self, cached_value)):
            cached_value = initialize_func(self)
            setattr(self, attr_name, cached_value)

        return cached_value

    # Use the original function's docstring
    getter.__doc__ = initialize_func.__doc__

    return property(getter)
