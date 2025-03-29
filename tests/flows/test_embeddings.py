import uuid
from unittest.mock import ANY, AsyncMock, MagicMock, patch  # Added AsyncMock

import chromadb
import pytest

# Mock the logger if it's not easily accessible or configured for tests
from buttermilk import logger as vector_logger  # Use alias to avoid conflict

# Assuming your module is structured like: buttermilk/data/vector.py
# Adjust the import path based on your project structure
from buttermilk.data.vector import (
    MODEL_NAME,
    ChromaDBEmbeddings,
    ChunkedDocument,
    InputDocument,  # Import default for type hinting if needed
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
    # Mock the synchronous get_embeddings method
    mock_model.get_embeddings.return_value = [mock_embedding_result]
    return mock_model


@pytest.fixture
def mock_chroma_collection():
    """Fixture for a mocked ChromaDB Collection instance."""
    mock_collection = MagicMock()
    # Mock the upsert method (it's synchronous in the chromadb client)
    mock_collection.upsert = MagicMock()
    return mock_collection


@pytest.fixture
def mock_async_citation_generator():
    """Fixture for a mocked async citation generator function."""
    # Use AsyncMock for async functions
    mock_generator = AsyncMock(return_value="Generated Citation: Test")
    return mock_generator


@pytest.fixture(autouse=True)  # Apply mocking automatically to relevant tests
def mock_dependencies(mocker, mock_text_embedding_model, mock_chroma_collection):
    """Mocks external dependencies like ChromaDB client and Vertex AI model loading."""
    # Mock Vertex AI model loading
    mocker.patch(
        "vertexai.language_models.TextEmbeddingModel.from_pretrained",
        return_value=mock_text_embedding_model,
    )
    # Mock ChromaDB client and collection retrieval
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_chroma_collection
    mocker.patch("chromadb.PersistentClient", return_value=mock_client)
    # Mock pdfminer text extraction
    mocker.patch(
        "pdfminer.high_level.extract_text",
        return_value="Paragraph 1.\n\nParagraph 2, which is longer.",
    )
    # Mock asyncio.to_thread to avoid actual threading in tests if _run_embedding_task is called directly
    # Often mocking higher-level methods like _embed makes this unnecessary
    mocker.patch("asyncio.to_thread", new_callable=AsyncMock)


@pytest.fixture
def input_doc_factory():
    """Factory fixture to create InputDocument instances."""

    def _factory(
        file_path="dummy/path.pdf",
        record_id="doc1",
        title="Test Doc",
        metadata=None,
    ):
        if metadata is None:
            metadata = {"doi": "10.1234/test", "url": "http://example.com"}
        # Ensure metadata is always a dict
        elif not isinstance(metadata, dict):
            metadata = {}
        return InputDocument(
            file_path=file_path,
            record_id=record_id,
            title=title,
            metadata=metadata.copy(),  # Return a copy
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
            metadata = {"doi": "10.1234/test", "url": "http://example.com"}
        # Ensure metadata is always a dict
        elif not isinstance(metadata, dict):
            metadata = {}
        return ChunkedDocument(
            chunk_id=chunk_id,
            document_title=doc_title,
            chunk_index=index,
            chunk_text=text,
            document_id=doc_id,
            embedding=embedding,
            metadata=metadata.copy(),  # Return a copy
        )

    return _factory


@pytest.fixture
def vector_store(
    tmp_path,
    mock_async_citation_generator,
    mock_text_embedding_model,
    mock_chroma_collection,
):
    """Fixture for a ChromaDBEmbeddings instance with mocked dependencies."""
    persist_dir = tmp_path / "chroma_test"
    persist_dir.mkdir()
    # Initialization triggers mocked load_models
    store = ChromaDBEmbeddings(
        collection_name="test_collection",
        persist_directory=str(persist_dir),
        chunk_size=50,  # Set a small chunk size for testing splitting
        chunk_overlap=5,
        citation_generator=mock_async_citation_generator,  # Inject mock generator
    )
    # Re-assign mocks as PrivateAttr might not be accessible otherwise post-init
    store._embedding_model = mock_text_embedding_model
    store._collection = mock_chroma_collection
    # Ensure the client mock is also available if needed directly
    store._client = chromadb.PersistentClient(
        path=str(persist_dir),
    )  # Get the mock client instance
    return store


# --- Test Cases ---


# Basic model tests remain synchronous
def test_input_document_creation():
    """Test basic InputDocument creation."""
    doc = InputDocument(
        file_path="a.pdf",
        record_id="r1",
        title="T1",
        metadata={"key": "value"},
    )
    assert doc.file_path == "a.pdf"
    assert doc.record_id == "r1"
    assert doc.title == "T1"
    assert doc.metadata == {"key": "value"}


def test_chunked_document_creation(chunked_doc_factory):
    """Test basic ChunkedDocument creation and defaults."""
    chunk = chunked_doc_factory(text="Test chunk content")
    assert isinstance(chunk.chunk_id, str)
    assert len(chunk.chunk_id) > 0
    assert chunk.document_title == "Test Doc"
    assert chunk.chunk_index == 0
    assert chunk.chunk_text == "Test chunk content"
    assert chunk.document_id == "doc1"
    assert chunk.embedding is None
    assert chunk.metadata == {"doi": "10.1234/test", "url": "http://example.com"}
    assert chunk.chunk_title == "Test Doc_0"


# Initialization test remains synchronous as validator runs during init
def test_vector_store_initialization(
    vector_store,
    mock_logger,
    tmp_path,
    mock_async_citation_generator,
):
    """Test that dependencies are loaded during initialization."""
    persist_dir = str(tmp_path / "chroma_test")
    # Check if mocks were called during initialization via the fixture
    vector_logger.info.assert_any_call(f"Loading embedding model: {MODEL_NAME}")
    vector_logger.info.assert_any_call(
        f"Initializing ChromaDB client at: {persist_dir}",
    )
    vector_logger.info.assert_any_call("Using ChromaDB collection: test_collection")

    # Verify mocks were called
    vector_store._embedding_model.from_pretrained.assert_called_once_with(MODEL_NAME)
    vector_store._client.PersistentClient.assert_called_once_with(path=persist_dir)
    vector_store._client.get_or_create_collection.assert_called_once_with(
        "test_collection",
    )
    assert vector_store.collection is not None
    # Check that the citation generator was assigned
    assert vector_store.citation_generator is mock_async_citation_generator


@patch(
    "pdfminer.high_level.extract_text",
    return_value="First paragraph.\n\nSecond paragraph, slightly longer.\n\nThird one.",
)
@pytest.mark.anyio
async def test_prepare_docs_basic(
    mock_extract,
    vector_store,
    input_doc_factory,
    mock_async_citation_generator,
):
    """Test basic document preparation, paragraph splitting, and citation generation."""
    input_doc = input_doc_factory(metadata={"original": "value"})
    full_text = mock_extract.return_value

    chunks = await vector_store.prepare_docs([input_doc])  # Await async method

    mock_extract.assert_called_once_with(input_doc.file_path, laparams=ANY)

    assert len(chunks) == 3
    assert chunks[0].chunk_text == "First paragraph."
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == input_doc.record_id

    assert chunks[1].chunk_text == "Second paragraph, slightly longer."
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_text == "Third one."
    assert chunks[2].chunk_index == 2


@patch(
    "pdfminer.high_level.extract_text",
    return_value="This is a single paragraph that is definitely longer than the chunk size limit set in the fixture.",
)
@pytest.mark.anyio
async def test_prepare_docs_splitting(
    mock_extract,
    vector_store,
    input_doc_factory,
    mock_async_citation_generator,
):
    """Test splitting of paragraphs larger than chunk_size, includes citation."""
    input_doc = input_doc_factory(metadata={"key": "val"})
    vector_store.chunk_size = 30  # Override for this test
    vector_store.chunk_overlap = 5
    full_text = mock_extract.return_value

    chunks = await vector_store.prepare_docs([input_doc])  # Await async method

    mock_extract.assert_called_once_with(input_doc.file_path, laparams=ANY)

    assert len(chunks) > 1  # Should be split
    assert chunks[0].chunk_text == "This is a single paragraph th"  # First 30 chars
    assert chunks[1].chunk_text.startswith(
        "graph that is definitely",
    )  # Starts after overlap
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert all(c.document_id == input_doc.record_id for c in chunks)
    # Check metadata propagation in split chunks
    expected_metadata = {"key": "val", "citation": "Generated Citation: Test"}
    assert all(c.metadata == expected_metadata for c in chunks)


@pytest.mark.anyio
async def test_prepare_docs_missing_path(vector_store, input_doc_factory, mock_logger):
    """Test skipping document if file_path is missing (async)."""
    input_doc = input_doc_factory(file_path=None)
    chunks = await vector_store.prepare_docs([input_doc])  # Await async method
    assert len(chunks) == 0
    mock_logger.warning.assert_called_with(
        f"Skipping record {input_doc.record_id} (index 0): missing file_path.",
    )


@patch("pdfminer.high_level.extract_text", side_effect=Exception("PDF read error"))
@pytest.mark.anyio
async def test_prepare_docs_extraction_error(
    mock_extract,
    vector_store,
    input_doc_factory,
    mock_logger,
):
    """Test skipping document if PDF extraction fails (async)."""
    input_doc = input_doc_factory()
    chunks = await vector_store.prepare_docs([input_doc])  # Await async method
    assert len(chunks) == 0
    mock_logger.error.assert_called_with(
        f"Error extracting text from PDF {input_doc.file_path}: PDF read error",
        exc_info=True,
    )


# Mock the internal _embed method for simplicity in testing higher-level embed methods
@patch("buttermilk.data.vector.ChromaDBEmbeddings._embed", new_callable=AsyncMock)
@pytest.mark.anyio
async def test_embed_records(mock_embed, vector_store, chunked_doc_factory):
    """Test embedding multiple ChunkedDocument objects (async)."""
    mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Mocked embeddings
    chunks = [
        chunked_doc_factory(index=0, text="text1"),
        chunked_doc_factory(index=1, text="text2"),
    ]

    embeddings = await vector_store.embed_records(chunks)  # Await async method

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Check that _embed was called with correctly formatted TextEmbeddingInput
    mock_embed.assert_awaited_once()
    call_args = mock_embed.call_args[0][0]  # Get the 'inputs' sequence
    assert len(call_args) == 2
    assert call_args[0].text == "text1"
    assert call_args[0].task_type == "RETRIEVAL_DOCUMENT"
    assert call_args[0].title == "Test Doc_0"
    assert call_args[1].text == "text2"
    assert call_args[1].task_type == "RETRIEVAL_DOCUMENT"
    assert call_args[1].title == "Test Doc_1"


@patch("buttermilk.data.vector.ChromaDBEmbeddings._embed", new_callable=AsyncMock)
@pytest.mark.anyio
async def test_embed_query(mock_embed, vector_store):
    """Test embedding a single query string (async)."""
    mock_embed.return_value = [[0.7, 0.8, 0.9]]  # Mocked embedding
    query = "What is the meaning of life?"

    embedding = await vector_store.embed_query(query)  # Await async method

    assert embedding == [0.7, 0.8, 0.9]

    # Check that _embed was called correctly
    mock_embed.assert_awaited_once()
    call_args = mock_embed.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].text == query
    assert call_args[0].task_type == "RETRIEVAL_QUERY"
    assert call_args[0].title is None


@patch("buttermilk.data.vector.ChromaDBEmbeddings._embed", new_callable=AsyncMock)
@pytest.mark.anyio
async def test_embed_query_failure(mock_embed, vector_store):
    """Test query embedding returning None on failure."""
    mock_embed.return_value = [None]  # Simulate embedding failure
    query = "A failing query"
    embedding = await vector_store.embed_query(query)
    assert embedding is None


# Test the internal _embed method's handling of failures
@pytest.mark.anyio
async def test_internal_embed_handling_failure(
    vector_store,
    mock_text_embedding_model,
    mock_logger,
):
    """Test that _embed returns None for failed tasks, preserving order."""
    # Configure the mock model's side effect for get_embeddings
    good_result = MagicMock()
    good_result.values = [1.0, 2.0]
    mock_text_embedding_model.get_embeddings.side_effect = [
        [good_result],  # First call succeeds
        Exception("API Error"),  # Second call fails
        [good_result],  # Third call succeeds
    ]

    # Mock asyncio.to_thread to directly call the side effect
    async def mock_to_thread(func, *args, **kwargs):
        try:
            # Simulate the behavior of calling the mocked get_embeddings
            return func(*args, **kwargs)
        except Exception as e:
            # Propagate the exception as the real to_thread would
            raise e

    with patch("asyncio.to_thread", mock_to_thread):
        inputs = [MagicMock(), MagicMock(), MagicMock()]  # Dummy inputs
        results = await vector_store._embed(inputs)

    assert len(results) == 3
    assert results[0] == [1.0, 2.0]
    assert results[1] is None  # Failure represented by None
    assert results[2] == [1.0, 2.0]
    mock_logger.error.assert_called_once_with(
        "Error getting embedding for input 1: API Error",
        exc_info=True,
    )
    mock_logger.info.assert_called_with(
        "Embedding process completed. Success: 2, Failed: 1.",
    )


@pytest.mark.anyio
async def test_get_embedded_records(vector_store, chunked_doc_factory, mock_logger):
    """Test assigning embeddings back to ChunkedDocument objects (async)."""
    chunks_in = [chunked_doc_factory(index=0), chunked_doc_factory(index=1)]
    # Mock the embed_records call within get_embedded_records
    # Simulate one success and one failure
    mock_embedding_results = [[1.0], None]
    with patch.object(
        vector_store,
        "embed_records",
        new_callable=AsyncMock,
        return_value=mock_embedding_results,
    ) as mock_embed:
        chunks_out = await vector_store.get_embedded_records(
            chunks_in,
        )  # Await async method

        mock_embed.assert_awaited_once_with(chunks_in)
        assert len(chunks_out) == 2
        # First chunk should have embedding
        assert chunks_out[0].embedding == [1.0]
        # Second chunk should have None embedding due to simulated failure
        assert chunks_out[1].embedding is None
        # Ensure original objects were modified
        assert chunks_in[0].embedding == [1.0]
        assert chunks_in[1].embedding is None
        # Check logs
        mock_logger.warning.assert_called_once_with(
            f"Embedding failed for chunk 1 (ID: {chunks_in[1].chunk_id}), skipping assignment.",
        )
        mock_logger.info.assert_called_with(
            "Finished assigning embeddings. Assigned: 1, Failed/Skipped: 1.",
        )


@pytest.mark.anyio
async def test_create_vectorstore_from_input_docs(
    vector_store,
    input_doc_factory,
    mock_chroma_collection,
    mock_logger,
    mock_async_citation_generator,
):
    """Test the full async pipeline starting from InputDocument objects."""
    input_docs = [
        input_doc_factory(record_id="doc1", metadata={"orig": "v1"}),
        input_doc_factory(record_id="doc2", metadata={"orig": "v2"}),
    ]
    # Mock intermediate steps
    mock_prepared_chunks = [
        ChunkedDocument(
            chunk_id="c1",
            document_title="Test Doc",
            chunk_index=0,
            chunk_text="P1",
            document_id="doc1",
            metadata={"orig": "v1", "citation": "Gen1"},
        ),
        ChunkedDocument(
            chunk_id="c2",
            document_title="Test Doc",
            chunk_index=1,
            chunk_text="P2",
            document_id="doc1",
            metadata={"orig": "v1", "citation": "Gen1"},
        ),
        ChunkedDocument(
            chunk_id="c3",
            document_title="Test Doc",
            chunk_index=0,
            chunk_text="P3",
            document_id="doc2",
            metadata={"orig": "v2", "citation": "Gen2"},
        ),
    ]
    # Simulate one embedding failure
    mock_embedded_chunks = [
        ChunkedDocument(
            chunk_id="c1",
            document_title="Test Doc",
            chunk_index=0,
            chunk_text="P1",
            document_id="doc1",
            metadata={"orig": "v1", "citation": "Gen1"},
            embedding=[1.0],
        ),
        ChunkedDocument(
            chunk_id="c2",
            document_title="Test Doc",
            chunk_index=1,
            chunk_text="P2",
            document_id="doc1",
            metadata={"orig": "v1", "citation": "Gen1"},
            embedding=None,
        ),  # Failed
        ChunkedDocument(
            chunk_id="c3",
            document_title="Test Doc",
            chunk_index=0,
            chunk_text="P3",
            document_id="doc2",
            metadata={"orig": "v2", "citation": "Gen2"},
            embedding=[3.0],
        ),
    ]
    # Mock the citation generator to return different values based on input if needed, or just use the default mock
    mock_async_citation_generator.side_effect = ["Gen1", "Gen2"]

    with (
        patch.object(
            vector_store,
            "prepare_docs",
            new_callable=AsyncMock,
            return_value=mock_prepared_chunks,
        ) as mock_prepare,
        patch.object(
            vector_store,
            "get_embedded_records",
            new_callable=AsyncMock,
            return_value=mock_embedded_chunks,
        ) as mock_get_embedded,
    ):
        count = await vector_store.create_vectorstore_chromadb(
            input_docs,
        )  # Await async method

        mock_prepare.assert_awaited_once_with(input_docs=input_docs)
        mock_get_embedded.assert_awaited_once_with(mock_prepared_chunks)
        # Verify filtering: only c1 and c3 should be upserted
        mock_chroma_collection.upsert.assert_called_once_with(
            ids=["c1", "c3"],
            embeddings=[[1.0], [3.0]],
            metadatas=[
                {
                    "document_title": "Test Doc",
                    "chunk_index": 0,
                    "document_id": "doc1",
                    "orig": "v1",
                    "citation": "Gen1",
                },
                {
                    "document_title": "Test Doc",
                    "chunk_index": 0,
                    "document_id": "doc2",
                    "orig": "v2",
                    "citation": "Gen2",
                },
            ],
            documents=["P1", "P3"],
        )
        assert count == 2  # Only 2 records upserted
        mock_logger.info.assert_any_call(
            "Processing 2 input documents: Chunking, citation generation, and embedding asynchronously...",
        )
        mock_logger.warning.assert_any_call(
            "Excluded 1 records due to missing/failed embeddings before upserting.",
        )
        mock_logger.info.assert_any_call(
            "Upserting 2 chunks into ChromaDB collection 'test_collection'...",
        )


@pytest.mark.anyio
async def test_create_vectorstore_from_pre_chunked(
    vector_store,
    chunked_doc_factory,
    mock_chroma_collection,
    mock_logger,
):
    """Test async pipeline with pre-chunked documents (need embedding)."""
    chunked_docs_no_embedding = [
        chunked_doc_factory(chunk_id="c1", index=0, text="P1", embedding=None),
        chunked_doc_factory(chunk_id="c2", index=1, text="P2", embedding=None),
    ]
    mock_embedded_chunks = [
        chunked_doc_factory(chunk_id="c1", index=0, text="P1", embedding=[1.0]),
        chunked_doc_factory(chunk_id="c2", index=1, text="P2", embedding=[2.0]),
    ]

    with (
        patch.object(
            vector_store,
            "prepare_docs",
            new_callable=AsyncMock,
        ) as mock_prepare,
        patch.object(
            vector_store,
            "get_embedded_records",
            new_callable=AsyncMock,
            return_value=mock_embedded_chunks,
        ) as mock_get_embedded,
    ):
        count = await vector_store.create_vectorstore_chromadb(
            chunked_docs_no_embedding,
        )  # Await

        mock_prepare.assert_not_awaited()  # Should not prepare docs
        mock_get_embedded.assert_awaited_once_with(chunked_docs_no_embedding)
        mock_chroma_collection.upsert.assert_called_once()  # Check details in previous test
        assert count == 2
        mock_logger.info.assert_any_call(
            "Using 2 pre-chunked documents, generating embeddings asynchronously...",
        )


@pytest.mark.anyio
async def test_create_vectorstore_from_pre_embedded(
    vector_store,
    chunked_doc_factory,
    mock_chroma_collection,
    mock_logger,
):
    """Test async pipeline with pre-chunked and pre-embedded documents."""
    chunked_docs_with_embedding = [
        chunked_doc_factory(chunk_id="c1", index=0, text="P1", embedding=[1.0]),
        chunked_doc_factory(chunk_id="c2", index=1, text="P2", embedding=[2.0]),
    ]

    with (
        patch.object(
            vector_store,
            "prepare_docs",
            new_callable=AsyncMock,
        ) as mock_prepare,
        patch.object(
            vector_store,
            "get_embedded_records",
            new_callable=AsyncMock,
        ) as mock_get_embedded,
    ):
        count = await vector_store.create_vectorstore_chromadb(
            chunked_docs_with_embedding,
        )  # Await

        mock_prepare.assert_not_awaited()
        mock_get_embedded.assert_not_awaited()  # Should not embed again
        mock_chroma_collection.upsert.assert_called_once()  # Check details in previous test
        assert count == 2
        mock_logger.info.assert_any_call(
            "Using 2 pre-chunked and pre-embedded documents.",
        )


@pytest.mark.anyio
async def test_create_vectorstore_empty_input(vector_store, mock_logger):
    """Test create_vectorstore_chromadb with empty input list (async)."""
    count = await vector_store.create_vectorstore_chromadb([])  # Await
    assert count == 0
    mock_logger.warning.assert_called_with(
        "No input data provided to create_vectorstore_chromadb.",
    )


@pytest.mark.anyio
async def test_create_vectorstore_invalid_input_type(vector_store, mock_logger):
    """Test create_vectorstore_chromadb with invalid input type (async)."""
    count = await vector_store.create_vectorstore_chromadb([
        {"invalid": "data"},
    ])  # Await
    assert count == 0
    mock_logger.error.assert_called_with(
        f"Invalid input type: {type({'invalid': 'data'})}. Expected InputDocument or ChunkedDocument.",
    )


@pytest.mark.anyio
async def test_create_vectorstore_upsert_error(
    vector_store,
    input_doc_factory,
    mock_chroma_collection,
    mock_logger,
):
    """Test handling of errors during ChromaDB upsert (async)."""
    input_docs = [input_doc_factory()]
    mock_chroma_collection.upsert.side_effect = Exception("DB connection failed")

    # Mocks for prepare/embed steps are needed even if we test the final upsert error
    mock_prepared_chunks = [
        ChunkedDocument(
            chunk_id="c1",
            document_title="Test Doc",
            chunk_index=0,
            chunk_text="P1",
            document_id="doc1",
            metadata={"citation": "Gen1"},
        ),
    ]
    mock_embedded_chunks = [
        ChunkedDocument(
            chunk_id="c1",
            document_title="Test Doc",
            chunk_index=0,
            chunk_text="P1",
            document_id="doc1",
            metadata={"citation": "Gen1"},
            embedding=[1.0],
        ),
    ]

    with (
        patch.object(
            vector_store,
            "prepare_docs",
            new_callable=AsyncMock,
            return_value=mock_prepared_chunks,
        ),
        patch.object(
            vector_store,
            "get_embedded_records",
            new_callable=AsyncMock,
            return_value=mock_embedded_chunks,
        ),
    ):
        count = await vector_store.create_vectorstore_chromadb(input_docs)  # Await

        assert count == 0
        mock_chroma_collection.upsert.assert_called_once()  # It was called
        mock_logger.error.assert_called_with(
            "Failed to upsert data into ChromaDB: DB connection failed",
            exc_info=True,
        )
