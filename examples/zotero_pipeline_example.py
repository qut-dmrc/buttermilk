"""Example showing how to use the new ZoteroSource + ZoteroDownloadProcessor pattern.

This demonstrates the clean separation:
1. ZoteroSource: Yields item IDs with metadata (fast, efficient)
2. ZoteroDownloadProcessor: Downloads and extracts full text (expensive operation)

The pipeline orchestrates them together with caching and error handling.
"""

import asyncio
from buttermilk.libs.zotero_v2 import ZoteroSource, ZoteroDownloadProcessor, VectorStoreExistenceFilter
from buttermilk.pipeline import PipelineOrchestrator
from buttermilk.data.vector import ChromaDBEmbeddings


async def main():
    """Run a Zotero pipeline with clean source/processor separation."""

    # Configuration
    library_id = "your_library_id"
    save_dir = "./data/zotero"

    # Optional: Set up vector store for deduplication
    vector_store = ChromaDBEmbeddings(
        collection_name="zotero_docs",
        # ... other config
    )

    # Create source with optional filter
    source = ZoteroSource(
        library_id=library_id,
        save_dir=save_dir,
        filter=VectorStoreExistenceFilter(vector_store),  # Skip items that already exist
        force_full_sync=False,  # Use incremental sync
        max_items=100,  # Optional limit for testing
    )

    # Create processor
    downloader = ZoteroDownloadProcessor(
        library_id=library_id,
        save_dir=save_dir,
    )

    # Set up pipeline
    pipeline = PipelineOrchestrator(
        pipeline_name="zotero_download",
        source=source,
        processors=[downloader],
        concurrency=5,  # Download 5 items concurrently
        enable_record_cache=True,  # Cache downloaded content
    )

    # Run pipeline
    async for record in pipeline():
        print(f"Downloaded: {record.record_id} - {record.metadata.get('title', 'Unknown')[:50]}")
        print(f"Content length: {len(record.content or '')} chars")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
