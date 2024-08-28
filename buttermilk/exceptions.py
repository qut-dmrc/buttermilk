class ProcessingFinished(Exception):
    """Jobs done."""

    pass


class FatalError(Exception):
    ## Something has gone horribly wrong and the process must terminate.
    pass


class RateLimit(Exception):
    """Rate limit exceeded."""

    pass


class NoMoreResults(Exception):
    """Time or results exceeded."""

    pass


class Delay(Exception):
    ## Wait before running again
    pass
