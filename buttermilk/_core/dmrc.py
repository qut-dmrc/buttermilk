
from __future__ import annotations

from .bm_init import BM
from .log import logger  # noqa

# This is a singleton pattern for the BM class.
# The bm variable is initialized to None and will be set to an instance of BM
# when the module is imported. This allows other modules to access the same
# instance of BM without creating a new one.
bm: BM | None = None  # Other modules import and use this directly after it's set
