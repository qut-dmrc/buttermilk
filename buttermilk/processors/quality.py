"""Quality filter processor for Buttermilk pipeline.

This processor analyzes document-level corruption and filters out documents
that exceed corruption thresholds. It should be placed between the chunking
and embedding steps to prevent corrupt documents from being vectorized.

The processor uses the text_quality module's is_document_corrupt() to
perform document-level analysis on all chunks and applies configurable
thresholds to determine if a document should be filtered.
"""

from typing import AsyncGenerator

from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.types import Record
from buttermilk.utils.text_quality import is_document_corrupt


class QualityFilterProcessor(BaseModel):
    """Filter corrupt documents from the pipeline based on quality analysis.

    This processor performs document-level quality analysis by examining all
    chunks of a document. Documents that exceed the corruption_threshold are
    filtered out (not yielded), preventing them from being embedded and stored.

    Features:
    - Document-level analysis (not chunk-level)
    - Configurable corruption threshold
    - Preserves clean documents unchanged
    - Logs filtering decisions for observability
    - Fail-fast: missing chunks attribute raises error

    Configuration:
        corruption_threshold: Percentage of chunks that must be corrupt to filter (0-100)
        pattern_threshold: Percentage of repetitive pattern to consider corrupt (0-100)

    Example:
        >>> processor = QualityFilterProcessor(
        ...     corruption_threshold=95.0,
        ...     pattern_threshold=80.0
        ... )
        >>> async for record in processor.process(record, processor_stage="quality"):
        ...     # Only clean documents are yielded
        ...     print(f"Processing {record.record_id}")
    """

    corruption_threshold: float = Field(
        default=95.0,
        description="Minimum corruption rate (%) to filter document. " "Documents with >= this percentage of corrupt chunks are filtered.",
    )

    pattern_threshold: float = Field(
        default=80.0,
        description="Minimum repetitive pattern rate (%) to consider corrupt. " "Currently used by verification logic, reserved for future use.",
    )

    async def process(self, record: Record, *, processor_stage: str, **kwargs) -> AsyncGenerator[Record, None]:
        """Process a record by analyzing document quality and filtering if corrupt.

        This processor expects the record to have a `chunks` attribute from a prior
        chunking stage (e.g., SemanticSplitter). It analyzes all chunks to calculate
        document-level corruption rate and filters documents exceeding the threshold.

        Args:
            record: Record with chunks attached
            processor_stage: Stage name for metadata tracking
            **kwargs: Additional keyword arguments (ignored)

        Yields:
            Record if document passes quality check (corruption < threshold)
            Nothing if document is filtered due to high corruption

        Raises:
            ValueError: If record has no chunks attribute (fail-fast)
        """
        # Fail-fast: Require chunks attribute from prior stage
        if not hasattr(record, "chunks") or record.chunks is None:
            raise ValueError(f"Record {record.record_id} has no chunks. " "QualityFilterProcessor requires chunks from prior chunking stage.")

        if len(record.chunks) == 0:
            logger.warning(
                "Record has empty chunks list, filtering out",
                record_id=record.record_id,
                processor_stage=processor_stage,
            )
            return

        # Extract chunk texts for quality analysis
        chunk_texts = []
        for chunk in record.chunks:
            # Handle both dict and object chunk formats
            if isinstance(chunk, dict):
                chunk_texts.append(chunk.get("chunk_text", ""))
            else:
                chunk_texts.append(getattr(chunk, "chunk_text", ""))

        # Perform document-level quality analysis
        quality_result = is_document_corrupt(chunk_texts, threshold=self.corruption_threshold)

        is_corrupt = quality_result["is_corrupt"]
        corruption_rate = quality_result["corruption_rate"]
        total_chunks = quality_result["total_chunks"]
        corrupted_chunks = quality_result["corrupted_chunks"]

        # Get document title for logging
        title = record.metadata.get("title", record.record_id) if record.metadata else record.record_id

        # Apply threshold filter
        if is_corrupt:
            # Filter out (do not yield) corrupt document
            logger.info(
                f"🚫 Filtering corrupt document: {title[:100]}",
                record_id=record.record_id,
                corruption_rate=f"{corruption_rate:.1f}%",
                corrupted_chunks=corrupted_chunks,
                total_chunks=total_chunks,
                threshold=f"{self.corruption_threshold:.1f}%",
                processor_stage=processor_stage,
            )
            return  # Do not yield - document is filtered

        # Document passes quality check
        logger.debug(
            f"✅ Document passed quality check: {title[:100]}",
            record_id=record.record_id,
            corruption_rate=f"{corruption_rate:.1f}%",
            corrupted_chunks=corrupted_chunks,
            total_chunks=total_chunks,
            processor_stage=processor_stage,
        )

        yield record
