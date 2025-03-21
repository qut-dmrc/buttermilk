class ProcessingFinished(Exception):
    """Jobs done."""


class FatalError(Exception):
    # Something has gone horribly wrong and the process must terminate.
    pass


class ProcessingError(Exception):
    """Something has gone mildly wrong."""


class RateLimit(Exception):
    """Rate limit exceeded."""


class NoMoreResults(Exception):
    """Time or results exceeded."""


class Delay(Exception):
    # Wait before running again
    pass
