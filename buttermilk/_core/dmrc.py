from __future__ import annotations

from .bm_init import BM

# This is a singleton pattern for the BM class.
# The bm variable is initialized to None and will be set to an instance of BM
# when the module is imported. This allows other modules to access the same
# instance of BM without creating a new one.
_bm_instance: BM = None  # Private storage  # no-qa


def get_bm() -> BM:
    """Return the singleton BM instance."""
    if _bm_instance is None:
        raise RuntimeError("BM singleton not initialized. Make sure CLI or nb.init() has been run.")
    return _bm_instance


def set_bm(instance: BM) -> None:
    """Set the singleton BM instance."""
    global _bm_instance
    _bm_instance = instance
