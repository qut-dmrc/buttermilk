"""Source wrapper for limiting records to first N.

This module provides a source wrapper that limits iteration to the first N records,
applied BEFORE any replication to ensure consistent sampling.
"""

from collections.abc import AsyncIterator
from typing import Any

from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord


class LimitingSource:
    """Async iterator wrapper that yields only the first N records from source.

    This wrapper sits at the source layer BEFORE ReplicatingSource to ensure
    the limit applies to original records, not replicated ones.

    Example:
        ```python
        # Original source has 1350 records
        original_source = get_storage(config)

        # Limit to 50 records FIRST
        limited_source = LimitingSource(original_source, limit=50)

        # Then replicate each record 5 times → 250 records total
        source = ReplicatingSource(limited_source, num_runs=5)
        ```
    """

    def __init__(self, source: Any, limit: int):
        """Initialize limiting source wrapper.

        Args:
            source: Source iterator that yields BaseRecord objects
            limit: Maximum number of records to yield

        Raises:
            ValueError: If limit < 1
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        self.source = source
        self.limit = limit

        logger.info(
            f"🔢 LimitingSource initialized with limit={limit}",
            limit=limit,
        )

    async def __aiter__(self) -> AsyncIterator[BaseRecord]:
        """Yield up to limit records from source.

        Yields:
            BaseRecord objects, up to self.limit total
        """
        source_iter = self.source if hasattr(self.source, "__anext__") else self.source.__aiter__()

        count = 0
        async for record in source_iter:
            if count >= self.limit:
                logger.info(
                    f"🔢 LimitingSource reached limit ({self.limit}), stopping",
                    limit=self.limit,
                    yielded=count,
                )
                break

            count += 1
            yield record

        logger.info(
            f"🔢 LimitingSource finished, yielded {count} records",
            limit=self.limit,
            yielded=count,
        )
