#!/usr/bin/env python3
"""Demo script to test ChromaDBSearchTool with real searches.

This script demonstrates how to use the ChromaDBSearchTool with the zot.yaml
configuration to search the prosocial Zotero collection.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import OmegaConf

from buttermilk import BM, logger
from buttermilk.tools.chromadb_search import ChromaDBSearchTool


async def main():
    """Run a demo search using ChromaDBSearchTool."""
    logger.info("=== ChromaDB Search Tool Demo ===")

    # Initialize Hydra and load configuration
    with hydra.initialize(version_base=None, config_path="../../conf"):
        cfg = hydra.compose(config_name="config", overrides=[
            "+storage=zot",  # Use the zot.yaml storage configuration
        ])

        resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        bm = BM(**resolved_cfg_dict["bm"])  # type: ignore # Assuming dict matches BM fields
        # Set the singleton BM instance
        from buttermilk import set_bm

        set_bm(bm)  # Set the Buttermilk instance using the singleton pattern
        # Extract storage config
        storage_cfg = OmegaConf.to_container(cfg.storage, resolve=True)
        logger.info("Using collection", collection_name=storage_cfg["collection_name"])
        logger.info("Embedding model", embedding_model=storage_cfg["embedding_model"])

        # Create and initialize the search tool
        search_tool = ChromaDBSearchTool(**storage_cfg)
        await search_tool.initialize()

        # Get collection info
        count = search_tool.collection.count()
        logger.info("Collection contains embeddings", count=count)

        # Perform searches
        queries = [
            "what are transaction costs",
            "prosocial behavior theory",
            "commons governance institutions",
            "collective action problems",
        ]

        for query in queries:
            logger.info("\n============================================================")
            logger.info("QUERY", query=query)
            logger.info("============================================================")

            try:
                results = await search_tool.search(query=query, n_results=3)

                if not results:
                    logger.info("No results found.")
                    continue

                for i, result in enumerate(results, 1):
                    logger.info("--- Result ---", result_num=i)
                    logger.info("Document", document_title=result.document_title or "Unknown")
                    logger.info("Document ID", document_id=result.document_id)
                    logger.info("Chunk ID", chunk_id=result.id)
                    logger.info("Score", score=f"{result.score:.4f}" if result.score else "N/A")

                    # Show content preview
                    content_preview = result.content[:300].replace("\n", " ")
                    logger.info("Content", content=f"{content_preview}...")

                    # Show some metadata
                    if result.metadata:
                        important_keys = ["chunk_index", "content_type", "embedding_model", "created_timestamp"]
                        metadata_preview = {k: v for k, v in result.metadata.items() if k in important_keys}
                        if metadata_preview:
                            logger.info("Metadata", metadata=metadata_preview)

            except Exception as e:
                logger.error("Search failed", error=e)

        # Demo filtering by metadata
        logger.info("\n============================================================")
        logger.info("FILTERED SEARCH: Looking for abstracts about 'institutions'")
        logger.info("============================================================")

        try:
            filtered_results = await search_tool.search(
                query="institutions",
                n_results=5,
                where={"content_type": "abstract"},
            )

            logger.info("Found results with content_type='abstract'", count=len(filtered_results))
            for result in filtered_results[:2]:  # Show first 2
                logger.info("Document", document_title=result.document_title)
                logger.info("Content preview", content=f"{result.content[:200]}...")

        except Exception as e:
            logger.error("Filtered search failed", error=e)


if __name__ == "__main__":
    asyncio.run(main())
