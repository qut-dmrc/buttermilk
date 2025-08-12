"""Integration tests for embedding generation via Gemini API and Vertex AI.

These tests are intentionally minimal and rely on the shared bm fixture from
tests/conftest.py to initialize runtime context. Authentication is via ADC
configured by bm; no API keys are required.

Related to gh issue #184
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from buttermilk.data.vector import GeminiEmbeddingFunction

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM

BM_TEST_GEMINI_EMBED_MODELS = [
    "text-embedding-004", "gemini-embedding-001", "text-embedding-005"]


@pytest.mark.parametrize("embedding_model", BM_TEST_GEMINI_EMBED_MODELS)
def test_gemini_embedding_function(bm: BM, embedding_model: str) -> None:
    """Use GeminiEmbeddingFunction to vectorise; verify basic shape."""
    # Keep it tiny to save quota and latency
    texts = [
        "A tiny test document.",
        "Another small input.",
    ]

    # Prefer a small dimensionality to reduce payloads; 128 is supported on text-embedding-004/005
    dim = 128
    ef = GeminiEmbeddingFunction(embedding_model=embedding_model, dimensionality=dim)

    embeddings = ef(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(len(v) == dim for v in embeddings)


@pytest.mark.parametrize("embedding_model", BM_TEST_GEMINI_EMBED_MODELS)
def test_vertex_embedding_minimal_compare(bm: BM, embedding_model: str) -> None:
    """Call Vertex AI TextEmbeddingModel using aiplatform and verify it returns vectors.
    """
    texts = [
        "A tiny test document.",
        "Another small input.",
    ]
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained(embedding_model)
    dim = 128
    inputs: list[str | TextEmbeddingInput] = [TextEmbeddingInput(text=t) for t in texts]
    results = model.get_embeddings(texts=inputs, auto_truncate=False, output_dimensionality=dim)

    # Convert to plain lists where needed
    vectors = [list(r.values) for r in results]
    assert len(vectors) == len(texts)
    assert all(len(v) == dim for v in vectors)
