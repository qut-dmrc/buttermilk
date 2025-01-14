import os
import argparse

from typing import Optional, Sequence
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from google.cloud import aiplatform
from chromadb.utils import embedding_functions

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
from pydantic import BaseModel, Field
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
    dimensionality: Optional[int] = None
    chunk_size: int = 4000
    chunk_overlap:int = 1000

    def get_embeddings(self, chunked_documents: Sequence[ChunkedDocument]) -> Sequence[ChunkedDocument]:
        model = TextEmbeddingModel.from_pretrained(self.embedding_model)
        kwargs = dict(output_dimensionality=self.dimensionality, auto_truncate=False)
        
        inputs = [TextEmbeddingInput(text=chunk.chunk_text, task_type=self.task, title=chunk.chunk_title) for chunk in chunked_documents]
        
        embeddings = []
        for chunk in inputs:
            # we could optimise by passing multiple chunks as a time, but we'd have to calculate the max input token to do that
            e = model.get_embeddings([chunk], **kwargs)
            embeddings.extend(e)
        
        for i, e in enumerate(embeddings):
            chunked_documents[i].embedding = e.values
            
        return chunked_documents
    
    
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


    def make_vectorstore_chromadb(self,
        collection_name: str,
        records: Sequence[RecordInfo],
        embeddings_path: str,
        persist_directory: str = ".chroma",
    ):
        records = pd.read_json(dataset, orient='records', lines=True)

        chunks = self.prepare_docs(records=records)
        
        embeddings = self.get_embeddings(chunks)

        df = pd.DataFrame(embeddings)
        upload_dataframe_json(data=df, uri=embeddings_path)

        # Instantiate a persistent chroma client in the persist_directory.
        client = chromadb.PersistentClient(path=persist_directory)

        collection = client.create_collection(name=collection_name)
        
        # Add data to ChromaDB
        for chunk in embeddings:
            metadata = dict(**chunk.structured_data)
            metadata.update({"chunk_index": chunk.chunk_index, "chunk_title": chunk.chunk_title })
            collection.add(
                ids=[chunk.chunk_id],
                embeddings=[chunk.embedding],
                metadatas=[metadata], 
                documents=[chunk.chunk_text],
            )
        return client, collection

