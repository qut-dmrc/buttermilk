"""Integration test for ChromaDBSearchTool with real ChromaDB and embeddings.

This test uses the actual zot.yaml storage configuration to perform real searches
against the prosocial Zotero collection.
"""

import asyncio

import pytest

from buttermilk import logger
from buttermilk.tools.chromadb_search import ChromaDBSearchTool, SearchResult


@pytest.mark.integration
class TestChromaDBSearchToolIntegration:
    """Integration tests for ChromaDBSearchTool with real ChromaDB instance."""

    @pytest.fixture
    async def search_tool(self, conf):
        """Create a ChromaDBSearchTool instance with zot.yaml configuration."""
        # Extract just the storage config
        storage_cfg = conf.flows.zot.agents.rag_zotero.tools.chromadb_search

        # Create the search tool with the storage configuration
        search_tool = ChromaDBSearchTool(**storage_cfg)

        # Initialize the tool (connects to ChromaDB)
        await search_tool.initialize()

        return search_tool

        # No cleanup needed as we're just reading

    @pytest.mark.asyncio
    async def test_search_transaction_costs(self, search_tool):
        """Test searching for 'transaction costs' in the Zotero collection."""
        # Perform the search
        query = "what are transaction costs"
        logger.info(f"Searching for: {query}")

        results = await search_tool.search(query=query, n_results=5)

        # Verify we got results
        assert isinstance(results, list)
        assert len(results) > 0, "Should have found at least one result"
        assert len(results) <= 5, "Should not exceed requested number of results"

        # Verify result structure
        for i, result in enumerate(results):
            assert isinstance(result, SearchResult)
            assert result.id, f"Result {i} should have an ID"
            assert result.content, f"Result {i} should have content"
            assert result.document_id, f"Result {i} should have a document ID"

            # Log the result for inspection
            logger.info(f"\nResult {i + 1}:")
            logger.info(f"  Document: {result.document_title or 'Unknown'}")
            logger.info(f"  Score: {result.score}")
            logger.info(f"  Content preview: {result.content[:200]}...")
            if result.metadata:
                logger.info(f"  Metadata keys: {list(result.metadata.keys())}")

    @pytest.mark.asyncio
    async def test_search_with_filter(self, search_tool):
        """Test searching with metadata filters."""
        # Search for transaction costs but filter by content type if available
        results = await search_tool.search(
            query="transaction costs",
            n_results=3,
            where={"content_type": "abstract"},  # Assuming abstracts are tagged
        )

        # Verify filtering worked (may return 0 results if no abstracts match)
        assert isinstance(results, list)
        assert len(results) <= 3

        # If we got results, verify they match the filter
        for result in results:
            if "content_type" in result.metadata:
                assert result.metadata["content_type"] == "abstract"

    @pytest.mark.asyncio
    async def test_search_no_duplicates(self, search_tool):
        """Test search with no_duplicates option."""
        # Configure tool to filter duplicates
        search_tool.no_duplicates = True

        results = await search_tool.search(
            query="economics institution",
            n_results=10,
        )

        # Verify no duplicate documents
        seen_docs = set()
        for result in results:
            assert result.document_id not in seen_docs, f"Found duplicate document: {result.document_id}"
            seen_docs.add(result.document_id)

    @pytest.mark.asyncio
    async def test_empty_query(self, search_tool):
        """Test behavior with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_tool.search(query="", n_results=5)

    @pytest.mark.asyncio
    async def test_collection_info(self, search_tool):
        """Test getting collection information."""
        # Get collection stats
        count = search_tool.collection.count()
        logger.info(f"Collection '{search_tool.collection_name}' contains {count} embeddings")

        assert count > 0, "Collection should not be empty"

        # Peek at some metadata to understand the collection
        peek_results = search_tool.collection.peek(limit=5)
        if peek_results and "metadatas" in peek_results:
            logger.info("\nSample metadata fields:")
            for i, metadata in enumerate(peek_results["metadatas"][:3]):
                logger.info(f"  Document {i + 1} metadata keys: {list(metadata.keys())}")

    @pytest.mark.asyncio
    async def test_tool_function_interface(self, search_tool):
        """Test the tool's function interface for agent integration."""
        # Get the autogen FunctionTool
        function_tool = search_tool.as_tool()

        assert function_tool is not None
        assert function_tool.name == "chromadb_search"
        assert function_tool.description

        # Test calling through the function interface
        result = await function_tool.run_json({"query": "prosocial theory", "n_results": 2})

        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) <= 2


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(pytest.main([__file__, "-v", "-s"]))
