import os
import argparse

from typing import Optional, Self, Sequence
import numpy as np
import pandas as pd
import pydantic
from tqdm import tqdm

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from google.cloud import aiplatform
from chromadb.utils import embedding_functions

from chromadb.utils.embedding_functions.chroma_langchain_embedding_function import (  # noqa: F401
    create_langchain_embedding,
)
from buttermilk._core.runner_types import RecordInfo
from buttermilk.utils.save import upload_dataframe_json
from buttermilk.utils.utils import read_text
from langchain_chroma import Chroma

import pandas as pd
from tqdm import tqdm

import chromadb
import google.generativeai as genai

from buttermilk.utils.utils import read_text
from langchain_chroma import Chroma

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from typing import Any, Mapping, Optional, Sequence
from pydantic import BaseModel, Field, PrivateAttr
from chromadb import Documents, Embeddings
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import uuid
import vertexai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from vertexai.language_models import TextEmbeddingModel

MODEL_NAME = "text-embedding-005"

class ChunkedDocument(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str
    document_title: str
    chunk_index: int
    chunk_text: str
    embedding: Optional[Sequence[float]|Sequence[int]] = None

    @property
    def chunk_title(self):
        return f"{self.document_title}_{self.chunk_index}"


class GoogleVertexEmbeddings(BaseModel):
    embedding_model: str = MODEL_NAME
    task: str = "RETRIEVAL_DOCUMENT"
    collection_name: str
    dimensionality: Optional[int] = None
    chunk_size: int = 4000
    chunk_overlap:int = 1000
    persist_directory: str = ".chroma"
    
    _collection = PrivateAttr(default=None)
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
    def get_collection(self) -> Any:
        self._collection = self._client.get_collection(self.collection_name, embedding_function=self._lc_embedding_function)

    @property
    def db(self) -> Any:
        db = Chroma(self.collection_name, persist_directory=self.persist_directory, embedding_function=self._lc_embedding_function, create_collection_if_not_exists=False)
        return db
    
    def embed_records(self, chunked_documents: Sequence[ChunkedDocument]) -> Embeddings:
        inputs = [TextEmbeddingInput(text=chunk.chunk_text, task_type=self.task, title=chunk.chunk_title) for chunk in chunked_documents]
        return self._embed(inputs)

    def embed_documents(self, texts: Documents) -> Embeddings:
        inputs = [TextEmbeddingInput(text=text, task_type=self.task) for text in texts]
        return self._embed(inputs)

    def embed_query(self, query: str) -> Any:
        inputs = [TextEmbeddingInput(text=query, task_type=self.task)]
        return self._embed(inputs)

    def _embed(self, inputs: Sequence[TextEmbeddingInput]) -> Embeddings:
        kwargs = dict(output_dimensionality=self.dimensionality, auto_truncate=False)
        embeddings = []
        for chunk in inputs:        
            # we could optimise by passing multiple chunks as a time, but we'd have to calculate the max input token to do that
            e = self._embedding_model.get_embeddings([chunk], **kwargs)
            embeddings.extend(e)

        embeddings = [ embedding.values  for embedding in embeddings]

        return embeddings


    def prepare_docs(self, records: Sequence[RecordInfo]) -> Sequence[ChunkedDocument]:
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        chunked_documents = []
        for record in records:
            chunk_texts = text_splitter.split_text(record.text)
            for i, chunk_text in enumerate(chunk_texts):              
                chunk = ChunkedDocument(record_id=record.record_id, document_title=record.title, chunk_index=i, chunk_text=chunk_text)
                chunked_documents.append(chunk)

        return chunked_documents


    def get_embedded_records(self, chunked_documents: Sequence[ChunkedDocument]) -> Sequence[ChunkedDocument]:
        
        embeddings = self.embed_records(chunked_documents)
        
        for i, embedding in enumerate(embeddings):
            chunked_documents[i].embedding = embedding
            
        return chunked_documents
    

    def create_vectorstore_chromadb(self,
                                  records: list[RecordInfo], 
                                  save_path: str,
    ) -> pd.DataFrame:
        
        docs = self.prepare_docs(records=records)
        embedded_records = self.get_embedded_records(docs)
        df_embeddings = pd.DataFrame.from_records([x.model_dump() for x in embedded_records])
        df_embeddings.to_json(save_path, orient='records', lines=True)

        ids = df_embeddings['chunk_id'].to_list()
        embeddings = df_embeddings['embedding'].to_list()
        metadata = df_embeddings[['record_id','document_title','chunk_index']].to_dict(orient='records')

        # Add all the embeddings we have currently
        self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
            )

        return df_embeddings