import uuid
from collections.abc import Sequence
from typing import Any, Self

import chromadb
import pandas as pd
import pydantic
from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions.chroma_langchain_embedding_function import (
    create_langchain_embedding,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import BaseModel, Field, PrivateAttr
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from buttermilk._core.runner_types import RecordInfo

MODEL_NAME = "text-embedding-005"


class ChunkedDocument(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_title: str
    chunk_index: int
    chunk_text: str
    document_id: str
    embedding: Sequence[float] | Sequence[int] | None = None

    @property
    def chunk_title(self):
        return f"{self.document_title}_{self.chunk_index}"


class GoogleVertexEmbeddings(BaseModel):
    embedding_model: str = MODEL_NAME
    task: str = "RETRIEVAL_DOCUMENT"
    collection_name: str
    dimensionality: int | None = None
    chunk_size: int = 4000
    chunk_overlap: int = 1000
    persist_directory: str = ".chroma"

    _collection = PrivateAttr(default=None)
    _db = PrivateAttr(default=None)
    _embedding_model = PrivateAttr(default=None)
    _client = PrivateAttr(default=None)
    _lc_embedding_function = PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def load_models(self) -> Self:
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model)

        # Instantiate a persistent chroma client in the persist_directory.
        self._client = chromadb.PersistentClient(path=self.persist_directory)

        self._lc_embedding_function = create_langchain_embedding(self)

        return self

    @property
    def collection(self) -> Any:
        if not self._collection:
            self._collection = self._client.get_collection(
                self.collection_name, embedding_function=self._lc_embedding_function
            )
        return self._collection

    @property
    def db(self) -> Any:
        if not self._db:
            self._db = Chroma(self.collection_name, client=self._client, persist_directory=self.persist_directory, embedding_function=self._lc_embedding_function, create_collection_if_not_exists=False)
        return self._db

    def embed_records(self, chunked_documents: Sequence[ChunkedDocument]) -> Embeddings:
        inputs = [
            TextEmbeddingInput(
                text=chunk.chunk_text, task_type=self.task, title=chunk.chunk_title
            )
            for chunk in chunked_documents
        ]
        return self._embed(inputs)

    def embed_documents(self, texts: Documents) -> Embeddings:
        inputs = [TextEmbeddingInput(text=text, task_type=self.task) for text in texts]
        return self._embed(inputs)

    def embed_query(self, query: str) -> Any:
        inputs = [TextEmbeddingInput(text=query, task_type=self.task)]
        # Don't return a list in this case, since we only passed in a single string
        return self._embed(inputs)[0]

    def _embed(self, inputs: Sequence[TextEmbeddingInput]) -> Embeddings:
        kwargs = dict(output_dimensionality=self.dimensionality, auto_truncate=False)
        embeddings = []
        for chunk in inputs:
            # we could optimise by passing multiple chunks as a time, but we'd have to calculate the max input token to do that
            e = self._embedding_model.get_embeddings([chunk], **kwargs)
            embeddings.extend(e)

        embeddings = [embedding.values for embedding in embeddings]

        return embeddings

    def prepare_docs(self, records: Sequence[RecordInfo]) -> Sequence[ChunkedDocument]:
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        chunked_documents = []
        for record in records:
            chunk_texts = text_splitter.split_text(record.all_text)
            for i, chunk_text in enumerate(chunk_texts):
                chunk = ChunkedDocument(record_id=record.record_id, document_title=record.title, chunk_index=i, chunk_text=chunk_text)
                chunked_documents.append(chunk)

        return chunked_documents

    def get_embedded_records(
        self, chunked_documents: Sequence[ChunkedDocument]
    ) -> Sequence[ChunkedDocument]:
        embeddings = self.embed_records(chunked_documents)

        for i, embedding in enumerate(embeddings):
            chunked_documents[i].embedding = embedding

        return chunked_documents

    def create_vectorstore_chromadb(
        self,
        records: list[RecordInfo],
        create_embeddings: bool = False,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        # create a new collection (fails if exists)
        # Note we don't update the main collection object in this method
        collection = self._client.create_collection(
            name=self.collection_name, get_or_create=False
        )

        if create_embeddings:
            # Create new vectors
            docs = self.prepare_docs(records=records)
            embedded_records = self.get_embedded_records(docs)
            df_embeddings = pd.DataFrame.from_records([
                x.model_dump() for x in embedded_records
            ])
            df_embeddings.to_json(save_path, orient="records", lines=True)
        elif len(records[-1].embedding) > 0:
            df_embeddings = pd.DataFrame.from_records([x.model_dump() for x in records])
        else:
            raise ValueError(
                "You must pass an 'embedding' field in the record list or create new embeddings."
            )

        ids = df_embeddings["record_id"].to_list()
        documents = df_embeddings["text"].to_list()
        embeddings = df_embeddings["embedding"].to_list()
        metadata = df_embeddings.drop(
            columns=["record_id", "text", "embedding", "ground_truth"]
        ).to_dict(orient="records")
        # remove nulls
        metadata = [{k: v for k, v in rec.items() if v} for rec in metadata]

        # Add all the embeddings we have currently
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
            documents=documents,
        )

        # Clear the collection object so that it gets recreated with the embedding function next time
        self._collection = None

        return df_embeddings
