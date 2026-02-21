"""Source wrapper for replicating records N times.

This module provides a source wrapper that yields multiple copies of each
source record, enabling reliability studies and repeated processing without
multiplying API calls through nested processors.

Usage:
    # Wrap any source to yield 5 copies of each record
    source = ReplicatingSource(original_source, num_runs=5)

Architecture:
    - Replication happens ONCE at the outermost source layer
    - Each replicated record flows through the pipeline exactly once
    - Processors see each record only once (no internal multiplication)
    - This prevents exponential API call growth from nested processors
"""

from typing import Any, AsyncIterator

from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord


class ReplicatingSource:
    """Async iterator wrapper that yields N copies of each source record.

    This wrapper sits at the source layer of the pipeline, replicating each
    record N times BEFORE it enters the processor chain. This ensures:

    1. Each record flows through pipeline exactly once (no processor multiplication)
    2. Consistent record_id tracking with run_index metadata
    3. Predictable API call counts (source_records × num_runs × processors)

    Attributes:
        source: Original source iterator (must support __aiter__)
        num_runs: Number of copies to yield for each source record

    Example:
        ```python
        # Original source yields 100 records
        original_source = get_storage(config)

        # Replicate each record 5 times → 500 records total
        source = ReplicatingSource(original_source, num_runs=5)

        # Pipeline processes 500 records (100 × 5)
        # Each of 6 LLM processors sees each record once → 3,000 API calls
        # NOT 15,000 calls from processor-level multiplication
        ```

    Metadata:
        Each replicated record gets metadata added (record_id is preserved):
        ```python
        record.metadata["replication"] = {
            "run_index": 0,        # Which run (0-4 for num_runs=5)
            "total_runs": 5,       # Total number of runs
        }
        ```
    """

    def __init__(self, source: Any, num_runs: int = 1):
        """Initialize replicating source wrapper.

        Args:
            source: Source iterator that yields BaseRecord objects
            num_runs: Number of times to replicate each record (default: 1, no replication)

        Raises:
            ValueError: If num_runs < 1
        """
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")

        self.source = source
        self.num_runs = num_runs

        logger.info(
            f"🔄 ReplicatingSource initialized with num_runs={num_runs}",
            num_runs=num_runs,
        )

    async def __aiter__(self) -> AsyncIterator[BaseRecord]:
        """Yield N copies of each source record.

        Yields:
            BaseRecord objects with replication metadata added
        """
        # Ensure we have an async iterator
<<<<<<< HEAD
        source_iter = self.source if hasattr(self.source, "__anext__") else self.source.__aiter__()
=======
        source_iter = (
            self.source
            if hasattr(self.source, "__anext__")
            else self.source.__aiter__()
        )
>>>>>>> origin/stable

        record_count = 0
        async for record in source_iter:
            record_count += 1
            original_record_id = getattr(record, "record_id", f"record_{record_count}")

            # Yield num_runs copies of this record
            for run_index in range(self.num_runs):
                # Add replication metadata (record_id stays unchanged)
                metadata = record.metadata.copy() if record.metadata else {}
                metadata["replication"] = {
                    "run_index": run_index,
                    "total_runs": self.num_runs,
                }

                # Create replicated record with metadata only (preserve original record_id)
<<<<<<< HEAD
                replicated_record = record.model_copy(update={"metadata": metadata})
=======
                replicated_record = record.model_copy(
                    update={"metadata": metadata}
                )
>>>>>>> origin/stable

                logger.debug(
                    f"🔄 Yielding replicated record {run_index + 1}/{self.num_runs}",
                    record_id=original_record_id,
                    run_index=run_index,
                )

                yield replicated_record
