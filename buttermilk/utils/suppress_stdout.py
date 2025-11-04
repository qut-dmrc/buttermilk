"""Utilities for suppressing stdout output in MCP server contexts.

MCP (Model Context Protocol) uses JSON-RPC over stdio, so any output to stdout
will break the protocol. This module provides utilities to ensure all output
goes to stderr instead.
"""

import os
import sys
import warnings
from contextlib import contextmanager


def redirect_stdout_to_stderr() -> None:
    """Globally redirect stdout to stderr.

    This is useful for MCP servers where stdout is reserved for JSON-RPC
    communication and any other output must go to stderr.

    This should be called as early as possible in your application,
    ideally before any imports that might print to stdout.

    Example:
        >>> from buttermilk.utils.suppress_stdout import redirect_stdout_to_stderr
        >>> redirect_stdout_to_stderr()
        >>> # Now all print() calls go to stderr
    """
    # Redirect Python's stdout to stderr
    sys.stdout = sys.stderr

    # Also redirect warnings to stderr (they go to stdout by default)
    warnings.simplefilter("default")  # Reset to default

    # Override the warning handler to use stderr
    def warning_on_stderr(message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that always writes to stderr."""
        sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warning_on_stderr


def suppress_stdout_completely() -> None:
    """Completely suppress stdout by redirecting it to /dev/null.

    This is more aggressive than redirect_stdout_to_stderr() and should
    only be used when you're absolutely sure no legitimate output should
    go to stdout.

    For MCP servers, redirect_stdout_to_stderr() is usually preferable
    as it allows you to still see output for debugging.

    Example:
        >>> from buttermilk.utils.suppress_stdout import suppress_stdout_completely
        >>> suppress_stdout_completely()
        >>> print("This will be silently discarded")
    """
    # Open /dev/null for writing
    devnull = open(os.devnull, "w")

    # Redirect stdout to /dev/null
    sys.stdout = devnull


@contextmanager
def stdout_redirected_to_stderr():
    """Context manager to temporarily redirect stdout to stderr.

    Useful when you need to ensure a specific code block doesn't write
    to stdout, but want to restore it afterwards.

    Example:
        >>> from buttermilk.utils.suppress_stdout import stdout_redirected_to_stderr
        >>> with stdout_redirected_to_stderr():
        ...     print("This goes to stderr")
        >>> print("This goes to stdout")
    """
    old_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        yield
    finally:
        sys.stdout = old_stdout


@contextmanager
def stdout_suppressed():
    """Context manager to temporarily suppress stdout completely.

    Useful when you need to silence noisy third-party libraries
    for a specific operation.

    Example:
        >>> from buttermilk.utils.suppress_stdout import stdout_suppressed
        >>> with stdout_suppressed():
        ...     print("This is discarded")
        >>> print("This prints normally")
    """
    old_stdout = sys.stdout
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        yield
    finally:
        devnull.close()
        sys.stdout = old_stdout


def configure_for_mcp() -> None:
    """Configure Buttermilk for MCP server usage.

    This function:
    1. Redirects stdout to stderr globally
    2. Configures warnings to go to stderr
    3. Sets environment variables to suppress third-party output

    Call this FIRST in your MCP server, before any other imports:

    Example:
        >>> # mcp_server.py
        >>> from buttermilk.utils.suppress_stdout import configure_for_mcp
        >>> configure_for_mcp()  # Call FIRST
        >>>
        >>> from buttermilk import init_async, logger
        >>> # ... rest of your MCP server code
    """
    # Redirect stdout to stderr
    redirect_stdout_to_stderr()

    # Set environment variables to suppress common third-party noise
    # Suppress TensorFlow warnings (if used)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress asyncio debug output
    os.environ["PYTHONASYNCIODEBUG"] = "0"

    # Configure Python warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
