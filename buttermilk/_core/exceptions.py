"""Custom exceptions used throughout the Buttermilk framework.

This module defines a set of custom exceptions to handle specific error
conditions and control flow scenarios encountered within Buttermilk operations.
These exceptions provide more semantic meaning than generic Python exceptions
and can be caught and handled specifically by different parts of the framework.
"""


class ProcessingFinished(Exception):
    """Signal exception indicating that all processing jobs are complete.

    This exception is typically used in loops or iterative processes to signal
    a graceful exit when there is no more work to be done. It's not necessarily
    an error condition but rather a control flow mechanism.
    """
    pass


class FatalError(Exception):
    """Indicates a critical error that prevents further processing.

    When this exception is raised, it usually means that the system has
    encountered an unrecoverable situation, and the current operation or
    the entire process should terminate.
    """
    pass


class ProcessingError(Exception):
    """Indicates a non-fatal error occurred during processing.

    This exception is used for errors that might disrupt a specific part of
    a task but do not necessarily require the entire application to halt.
    It signifies that something went wrong, but the system might be able to
    recover, retry, or continue with other tasks.
    """
    pass


class RateLimit(Exception):
    """Indicates that an API rate limit has been exceeded.

    This exception is raised when an external service (e.g., an LLM API,
    a cloud provider API) reports that too many requests have been made
    within a certain time window. Handling this typically involves waiting
    for a period and then retrying the request.
    """
    pass


class NoMoreResults(Exception):
    """Signal exception indicating that no more results are available or expected.

    Similar to `ProcessingFinished`, this can be used to control loops that
    fetch or process paginated data or data from a queue. It signifies that
    the data source is exhausted or a predefined limit (e.g., time, number
    of results) has been reached.
    """
    pass


class Delay(Exception):
    """Signal exception indicating that a delay is required before proceeding.

    This is not an error but a control flow mechanism, often used in conjunction
    with retry logic or when waiting for an external process to complete.
    The component catching this exception should pause for a specified duration
    before attempting to run the operation again.
    """
    pass
