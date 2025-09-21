import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Mock the logger if it's not easily accessible or configured for tests
# Assuming your module is structured like: buttermilk/data/vector.py
# Adjust the import path based on your project structure
from buttermilk.data.vector import (
    MODEL_NAME,
    ChromaDBEmbeddings,
    ChunkedDocument,
    InputDocument,
    _batch_iterator,  # Import helper for testing if needed
)

# --- Fixtures ---

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_logger():
    """Fixture to mock the logger used in the vector module."""
    with patch("buttermilk.data.vector.logger", autospec=True) as mock_log:
        yield mock_log


@pytest.fixture
def mock_text_embedding_model():
    """Fixture for a mocked TextEmbeddingModel instance."""
    mock_model = MagicMock()
    # Simulate the structure returned by get_embeddings
    mock_embedding_result = MagicMock()
    mock_embedding_result.values = [0.1, 0.2, 0.3]  # Example embedding
    # Mock the async get_embeddings method
    mock_model.get_embeddings = AsyncMock(return_value=[mock_embedding_result])
    return mock_model


@pytest.fixture
def mock_chroma_collection():
    """Fixture for a mocked ChromaDB Collection instance."""
    mock_collection = MagicMock()
    # Mock the upsert method (it's synchronous in the chromadb client)
    mock_collection.upsert = MagicMock()
    return mock_collection


@pytest.fixture
def mock_text_splitter():
    """Fixture for a mocked RecursiveCharacterTextSplitter."""
    mock_splitter = MagicMock(spec=RecursiveCharacterTextSplitter)
    # Default behavior: return a list of simple chunks
    mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
    return mock_splitter


@pytest.fixture(autouse=True)  # Apply mocking automatically to relevant tests
def mock_dependencies(
    mocker,
    mock_text_embedding_model,
    mock_chroma_collection,
    mock_text_splitter,
):
    """Mocks external dependencies like ChromaDB client, Vertex AI model loading, and text splitter."""
    # Mock Vertex AI model loading
    mocker.patch(
        "vertexai.language_models.TextEmbeddingModel.from_pretrained",
        return_value=mock_text_embedding_model,
    )
    # Mock ChromaDB client and collection retrieval
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_chroma_collection
    mocker.patch("chromadb.PersistentClient", return_value=mock_client)

    # Mock the RecursiveCharacterTextSplitter instantiation within the class
    # This ensures our mock_text_splitter instance is used by the vector_store
    mocker.patch(
        "buttermilk.data.vector.RecursiveCharacterTextSplitter",
        return_value=mock_text_splitter,
    )

    # Mock asyncio.to_thread used for upsert
    mocker.patch("asyncio.to_thread", new_callable=AsyncMock)


@pytest.fixture
def input_doc_factory():
    """Factory fixture to create InputDocument instances."""

    def _factory(
        record_id="doc1",
        title="Test Doc",
        full_text="Default full text.",  # Added full_text
        metadata=None,
        file_path="dummy/path.pdf",  # Keep file_path for now if needed elsewhere
    ):
        if metadata is None:
            metadata = {"doi": "10.1234/test"}
        elif not isinstance(metadata, dict):
            metadata = {}
        return InputDocument(
            record_id=record_id,
            title=title,
            full_text=full_text,
            metadata=metadata.copy(),
            file_path=file_path,
        )

    return _factory


@pytest.fixture
def chunked_doc_factory():
    """Factory fixture to create ChunkedDocument instances."""

    def _factory(
        chunk_id=None,
        doc_title="Test Doc",
        index=0,
        text="Chunk text",
        doc_id="doc1",
        embedding=None,
        metadata=None,
    ):
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {"doi": "10.1234/test"}
        elif not isinstance(metadata, dict):
            metadata = {}
        return ChunkedDocument(
            chunk_id=chunk_id,
            document_title=doc_title,
            chunk_index=index,
            chunk_text=text,
            document_id=doc_id,
            embedding=embedding,
            metadata=metadata.copy(),
        )

    return _factory


@pytest.fixture
def vector_store(
    tmp_path,
    mock_text_embedding_model,
    mock_chroma_collection,
    mock_text_splitter,  # Inject the mock splitter fixture
):
    """Fixture for a ChromaDBEmbeddings instance with mocked dependencies."""
    persist_dir = tmp_path / "chroma_test"
    persist_dir.mkdir()
    # Initialization triggers mocked load_models and splitter creation
    store = ChromaDBEmbeddings(
        name="test",
        type="chromadb",
        collection_name="test_collection",
        persist_directory=str(persist_dir),
        concurrency=5,  # Example concurrency
        upsert_batch_size=50,  # Example upsert batch size
        embedding_batch_size=5,  # Example embedding batch size
    )
    # Re-assign mocks as PrivateAttr might not be accessible otherwise post-init
    store._embedding_model = mock_text_embedding_model
    store._collection = mock_chroma_collection
    # Ensure the client mock is also available if needed directly
    store._client = chromadb.PersistentClient(
        path=str(persist_dir),
    )  # Get the mock client instance
    return store


# Helper to convert list to async iterator for tests
async def list_to_async_iterator(items):
    for item in items:
        yield item
        await asyncio.sleep(0)  # Yield control briefly


# Helper to collect results from an async iterator
async def collect_async_iterator(aiter):
    return [item async for item in aiter]


# --- Test Cases ---


# Initialization test updated for text splitter
def test_vector_store_initialization(
    vector_store,
    mock_logger,
    tmp_path,
    mock_text_splitter,  # Add mock splitter
):
    """Test that dependencies are loaded during initialization."""
    persist_dir = str(tmp_path / "chroma_test")
    # Check if mocks were called during initialization via the fixture
    mock_logger.info.assert_any_call(f"Loading embedding model: {MODEL_NAME}")
    mock_logger.info.assert_any_call(f"Initializing ChromaDB client at: {persist_dir}")
    mock_logger.info.assert_any_call("Using ChromaDB collection: test_collection")
    # Check splitter initialization log
    mock_logger.info.assert_any_call(
        f"Initialized RecursiveCharacterTextSplitter (chunk_size={vector_store.chunk_size}, chunk_overlap={vector_store.chunk_overlap})",
    )

    # Verify mocks were called
    vector_store._embedding_model.from_pretrained.assert_called_once_with(MODEL_NAME)
    vector_store._client.PersistentClient.assert_called_once_with(path=persist_dir)
    vector_store._client.get_or_create_collection.assert_called_once_with(
        "test_collection",
    )
    # Verify the splitter mock was called during init
    vector_store._text_splitter.__init__.assert_called_once_with(
        chunk_size=vector_store.chunk_size,
        chunk_overlap=vector_store.chunk_overlap,
        length_function=len,
        add_start_index=False,
    )
    assert vector_store.collection is not None


# --- Tests for prepare_docs (Async Generator) ---


@pytest.mark.anyio
async def test_prepare_docs_uses_splitter_and_yields(
    vector_store,
    input_doc_factory,
    mock_text_splitter,
):
    """Test that prepare_docs uses the text splitter and yields ChunkedDocuments."""
    input_text = "This is the input text to be split."
    input_doc = input_doc_factory(full_text=input_text, metadata={"orig": "val"})
    # Configure mock splitter for this test
    mock_text_splitter.split_text.return_value = ["Chunk 1 text", "Chunk 2 text"]

    input_iterator = list_to_async_iterator([input_doc])
    output_chunks = await collect_async_iterator(
        vector_store.prepare_docs(input_iterator),
    )

    # Verify splitter was called with the correct text
    mock_text_splitter.split_text.assert_called_once_with(input_text)

    # Verify output
    assert len(output_chunks) == 2
    assert isinstance(output_chunks[0], ChunkedDocument)
    assert output_chunks[0].chunk_text == "Chunk 1 text"
    assert output_chunks[0].chunk_index == 0
    assert output_chunks[0].document_id == input_doc.record_id
    assert output_chunks[0].metadata == {"orig": "val"}  # Metadata copied

    assert isinstance(output_chunks[1], ChunkedDocument)
    assert output_chunks[1].chunk_text == "Chunk 2 text"
    assert output_chunks[1].chunk_index == 1
    assert output_chunks[1].document_id == input_doc.record_id
    assert output_chunks[1].metadata == {"orig": "val"}


@pytest.mark.anyio
async def test_prepare_docs_empty_full_text(
    vector_store,
    input_doc_factory,
    mock_logger,
    mock_text_splitter,
):
    """Test that prepare_docs skips documents with empty full_text."""
    input_doc = input_doc_factory(full_text="")
    input_iterator = list_to_async_iterator([input_doc])
    output_chunks = await collect_async_iterator(
        vector_store.prepare_docs(input_iterator),
    )

    assert len(output_chunks) == 0
    mock_text_splitter.split_text.assert_not_called()
    mock_logger.warning.assert_called_with(
        f"Skipping record {input_doc.record_id} (document #1): missing full_text.",
    )


@pytest.mark.anyio
async def test_prepare_docs_splitter_error(
    vector_store,
    input_doc_factory,
    mock_logger,
    mock_text_splitter,
):
    """Test that prepare_docs handles errors during text splitting."""
    input_doc = input_doc_factory(full_text="Some text")
    mock_text_splitter.split_text.side_effect = Exception("Splitter failed")

    input_iterator = list_to_async_iterator([input_doc])
    output_chunks = await collect_async_iterator(
        vector_store.prepare_docs(input_iterator),
    )

    assert len(output_chunks) == 0
    mock_text_splitter.split_text.assert_called_once_with("Some text")
    mock_logger.error.assert_called_with(
        f"Error splitting text for doc {input_doc.record_id}: Splitter failed",
        exc_info=True,
    )


# --- Tests for get_embedded_records (Async Generator) ---


@pytest.mark.anyio
async def test_get_embedded_records_yields_embedded_chunks(
    vector_store,
    chunked_doc_factory,
    mock_logger,
):
    """Test that get_embedded_records yields chunks with embeddings."""
    chunk1 = chunked_doc_factory(chunk_id="c1", index=0, text="Text 1")
    chunk2 = chunked_doc_factory(chunk_id="c2", index=1, text="Text 2")
    input_iterator = list_to_async_iterator([chunk1, chunk2])

    # Mock the internal embed_records call
    mock_embeddings = [[1.0], [2.0]]
    with patch.object(vector_store, "embed_records", new_callable=AsyncMock, return_value=mock_embeddings) as mock_embed:
        output_chunks = await collect_async_iterator(
            vector_store.get_embedded_records(input_iterator, batch_size=2),
        )

        # Verify embed_records was called (likely once for the batch)
        mock_embed.assert_awaited_once()
        # Check the batch passed to embed_records
        assert mock_embed.await_args[0][0] == [chunk1, chunk2]

        # Verify output
        assert len(output_chunks) == 2
        assert output_chunks[0].chunk_id == "c1"
        assert output_chunks[0].embedding == [1.0]
        assert output_chunks[1].chunk_id == "c2"
        assert output_chunks[1].embedding == [2.0]


@pytest.mark.anyio
async def test_get_embedded_records_skips_failed_embeddings(
    vector_store,
    chunked_doc_factory,
    mock_logger,
):
    """Test that get_embedded_records skips yielding chunks where embedding failed."""
    chunk1 = chunked_doc_factory(chunk_id="c1", index=0, text="Text 1")
    chunk2 = chunked_doc_factory(
        chunk_id="c2",
        index=1,
        text="Text 2",
    )  # This one will fail
    chunk3 = chunked_doc_factory(chunk_id="c3", index=2, text="Text 3")
    input_iterator = list_to_async_iterator([chunk1, chunk2, chunk3])

    # Mock the internal embed_records call - simulate one failure
    mock_embeddings = [[1.0], None, [3.0]]
    with patch.object(vector_store, "embed_records", new_callable=AsyncMock, return_value=mock_embeddings) as mock_embed:
        # Use a batch size smaller than the total number of chunks to test batching log messages
        output_chunks = await collect_async_iterator(
            vector_store.get_embedded_records(input_iterator, batch_size=2),
        )

        # Verify embed_records was called (likely twice for batches of 2)
        assert mock_embed.await_count == 2

        # Verify output - only chunks with successful embeddings should be yielded
        assert len(output_chunks) == 2
        assert output_chunks[0].chunk_id == "c1"
        assert output_chunks[0].embedding == [1.0]
        assert output_chunks[1].chunk_id == "c3"
        assert output_chunks[1].embedding == [3.0]

        # Check warning log for the skipped chunk
        mock_logger.warning.assert_called_with(
            f"Embedding failed for chunk 1 (ID: {chunk2.chunk_id}), skipping.",
        )
        # Check final log message
        mock_logger.info.assert_any_call(
            "Finished embedding process. Total Chunks Processed: 3, Succeeded: 2, Failed: 1.",
        )


# --- Tests for create_vectorstore_chromadb (Pipeline Orchestration) ---


@pytest.mark.anyio
async def test_create_vectorstore_pipeline(
    vector_store,
    input_doc_factory,
    chunked_doc_factory,
    mock_chroma_collection,
    mock_logger,
):
    """Test the full pipeline orchestration in create_vectorstore_chromadb."""
    # Input data
    input_doc1 = input_doc_factory(record_id="d1", full_text="Text one.")
    input_doc2 = input_doc_factory(record_id="d2", full_text="Text two.")
    input_iterator = list_to_async_iterator([input_doc1, input_doc2])

    # Mock outputs of pipeline stages
    prepared_chunk1 = chunked_doc_factory(doc_id="d1", index=0, text="Chunk 1.1")
    prepared_chunk2 = chunked_doc_factory(doc_id="d2", index=0, text="Chunk 2.1")
    prepared_chunk3 = chunked_doc_factory(doc_id="d2", index=1, text="Chunk 2.2")

    embedded_chunk1 = chunked_doc_factory(doc_id="d1", index=0, text="Chunk 1.1", embedding=[1.0])
    # Simulate embedding failure for prepared_chunk2
    embedded_chunk3 = chunked_doc_factory(doc_id="d2", index=1, text="Chunk 2.2", embedding=[3.0])

    # Mock the generator methods
    async def mock_prepare_gen(*args, **kwargs):
        yield prepared_chunk1
        yield prepared_chunk2
        yield prepared_chunk3
        await asyncio.sleep(0)

    async def mock_embed_gen(*args, **kwargs):
        # Simulate filtering based on embedding success
        yield embedded_chunk1
        # Skip chunk 2
        yield embedded_chunk3
        await asyncio.sleep(0)

    with (
        patch.object(
            vector_store,
            "prepare_docs",
            side_effect=mock_prepare_gen,
        ) as mock_prepare,
        patch.object(
            vector_store,
            "get_embedded_records",
            side_effect=mock_embed_gen,
        ) as mock_embed,
        patch(
            "buttermilk.data.vector._batch_iterator",
            wraps=_batch_iterator,
        ) as mock_batch_iter,  # Wrap to test batching
    ):
        # Set a small upsert batch size to test batching
        vector_store.upsert_batch_size = 1
        count = await vector_store.create_vectorstore_chromadb(input_iterator)

        # Verify pipeline calls
        mock_prepare.assert_called_once()
        # Check the input to prepare_docs was the original iterator
        assert mock_prepare.call_args[0][0] is input_iterator

        mock_embed.assert_called_once()
        # Check the input to get_embedded_records was the output of prepare_docs
        # (Difficult to assert directly on the generator object, rely on flow)
        assert mock_embed.call_args[1]["batch_size"] == vector_store.concurrency  # Check batch size used

        # Verify batching for upsert (expect 2 batches of size 1)
        assert mock_batch_iter.call_count >= 1  # Called at least for the upsert stage
        upsert_batch_args = [call for call in mock_batch_iter.call_args_list if call[0][1] == vector_store.upsert_batch_size]
        assert len(upsert_batch_args) == 1  # Should be called once for the upsert stage batching

        # Verify upsert calls (should be 2 calls due to batch size 1)
        assert mock_chroma_collection.upsert.call_count == 2
        upsert_calls = vector_store._client.mock_calls  # Access upsert calls via mocked client->collection
        # Call 1
        call1_args = upsert_calls[0].args  # Adjust index if other client methods were mocked/called
        assert call1_args[0]["ids"] == [embedded_chunk1.chunk_id]
        assert call1_args[0]["embeddings"] == [embedded_chunk1.embedding]
        # Call 2
        call2_args = upsert_calls[1].args
        assert call2_args[0]["ids"] == [embedded_chunk3.chunk_id]
        assert call2_args[0]["embeddings"] == [embedded_chunk3.embedding]

        # Verify final count
        assert count == 2  # embedded_chunk1 and embedded_chunk3 were upserted

        # Verify logs
        mock_logger.info.assert_any_call(
            f"Starting vector store creation pipeline for collection '{vector_store.collection_name}' (upsert batch size: {vector_store.upsert_batch_size})."
        )
        mock_logger.info.assert_any_call("Upserting batch #1 (1 chunks) into ChromaDB collection 'test_collection'...")
        mock_logger.info.assert_any_call("Upserting batch #2 (1 chunks) into ChromaDB collection 'test_collection'...")
        mock_logger.info.assert_any_call("Vector store creation pipeline finished. Total chunks successfully upserted: 2.")


@pytest.mark.anyio
async def test_create_vectorstore_empty_iterator(vector_store, mock_logger):
    """Test create_vectorstore with an empty input iterator."""
    input_iterator = list_to_async_iterator([])

    with (
        patch.object(vector_store, "prepare_docs") as mock_prepare,
        patch.object(vector_store, "get_embedded_records") as mock_embed,
    ):

        async def mock_empty_prepare(*args, **kwargs):
            if False:  # Never yield
                yield
            await asyncio.sleep(0)

        mock_prepare.side_effect = mock_empty_prepare

        count = await vector_store.create_vectorstore_chromadb(input_iterator)

        assert count == 0
        mock_prepare.assert_called_once()
        mock_embed.assert_not_called()  # Should not be called if prepare yields nothing
        vector_store._collection.upsert.assert_not_called()
        # Check for the final log message indicating 0 upserts
        mock_logger.info.assert_any_call("Vector store creation pipeline finished. Total chunks successfully upserted: 0.")


@pytest.mark.anyio
async def test_create_vectorstore_upsert_error_handling(
    vector_store,
    input_doc_factory,
    chunked_doc_factory,
    mock_chroma_collection,
    mock_logger,
):
    """Test that the pipeline continues and logs errors if one upsert batch fails."""
    input_iterator = list_to_async_iterator([input_doc_factory()])
    embedded_chunk1 = chunked_doc_factory(embedding=[1.0], chunk_id="c1")
    embedded_chunk2 = chunked_doc_factory(embedding=[2.0], chunk_id="c2")

    async def mock_prepare(*_a, **_kw):
        yield chunked_doc_factory()
        yield chunked_doc_factory()  # Dummy prepare

    async def mock_embed(*_a, **_kw):
        yield embedded_chunk1
        yield embedded_chunk2  # Dummy embed

    # Simulate failure on the first upsert call, success on the second
    mock_chroma_collection.upsert.side_effect = [Exception("DB Write Error"), None]

    with (
        patch.object(vector_store, "prepare_docs", side_effect=mock_prepare),
        patch.object(vector_store, "get_embedded_records", side_effect=mock_embed),
        patch(
            "asyncio.to_thread",
            wraps=asyncio.to_thread,
        ),  # Use real to_thread but allow mocking collection inside
    ):
        vector_store.upsert_batch_size = 1  # Ensure two batches
        count = await vector_store.create_vectorstore_chromadb(input_iterator)

        # Verify upsert was attempted twice
        assert mock_chroma_collection.upsert.call_count == 2
        # Verify error was logged for the first batch
        mock_logger.error.assert_called_with(
            "Failed to upsert batch #1 into ChromaDB: DB Write Error",
            exc_info=True,
        )
        # Verify the second batch was still processed and logged
        mock_logger.info.assert_any_call("Upserting batch #2 (1 chunks) into ChromaDB collection 'test_collection'...")
        # Verify final count reflects only the successful upsert
        assert count == 1
        mock_logger.info.assert_any_call("Vector store creation pipeline finished. Total chunks successfully upserted: 1.")


# Remove or adapt old tests that used list-based processing if they are now obsolete
# e.g., test_create_vectorstore_from_input_docs, test_create_vectorstore_from_pre_chunked, etc.
# might be fully replaced by test_create_vectorstore_pipeline.
