import asyncio
import json
from pathlib import Path

import hydra
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm

from buttermilk import logger
from buttermilk.data.vector import (
    ChromaDBEmbeddings,
    ChunkedDocument,
    InputDocument,
    list_to_async_iterator,
)


async def import_parquet_files_to_chroma(
    vector_store: ChromaDBEmbeddings,
    parquet_dir=None,
    concurrency=10,
    batch_size=20,
):
    """Import parquet files to ChromaDB with parallel batch document upserts.

    Args:
        vector_store: Initialized ChromaDBEmbeddings instance
        parquet_dir: Directory containing parquet files (defaults to arrow_save_dir)
        concurrency: Max number of files to process in parallel
        batch_size: Number of documents to batch together for upserting

    """
    # Use the configured arrow_save_dir if no parquet_dir provided
    parquet_dir = parquet_dir or vector_store.arrow_save_dir
    parquet_dir_path = Path(parquet_dir)

    # Get all parquet files
    parquet_files = list(parquet_dir_path.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to process")

    # Stats counters
    successful_docs = 0
    failed_docs = 0
    total_chunks = 0

    # Semaphore to control concurrency
    sem = asyncio.Semaphore(concurrency)

    async def process_batch(batch_files):
        nonlocal successful_docs, failed_docs, total_chunks

        # Read batch of parquet files and convert to InputDocument objects
        loaded_docs = []

        for file_path in batch_files:
            async with sem:  # Control concurrent file reads
                try:
                    # Read the parquet file (using thread pool for file I/O)
                    table = await asyncio.to_thread(pq.read_table, file_path)

                    # Extract document metadata from schema
                    meta_dict = {}
                    for k, v in table.schema.metadata.items():
                        if k != b"pandas":  # Skip pandas metadata
                            try:
                                key = k.decode("utf-8")
                                value = v.decode("utf-8")
                                meta_dict[key] = value
                            except:
                                pass

                    # Convert to pandas for easy processing
                    df = table.to_pandas()

                    if len(df) == 0:
                        logger.warning(f"Empty dataframe in {file_path}, skipping")
                        continue

                    # Create InputDocument
                    doc = InputDocument(
                        record_id=meta_dict.get("record_id", file_path.stem),
                        file_path=meta_dict.get("file_path", ""),
                        record_path=meta_dict.get("record_path", ""),
                        chunks_path=str(file_path),
                        title=meta_dict.get("title", ""),
                        metadata=json.loads(meta_dict.get("metadata", "{}")),
                        chunks=[],
                    )

                    # Add chunks to document
                    for _, row in df.iterrows():
                        if row["embedding"] is None:
                            continue

                        # Convert NumPy array to Python list if needed
                        embedding = row["embedding"]
                        if hasattr(embedding, "tolist"):  # Check if it's a numpy array
                            embedding = embedding.tolist()

                        chunk = ChunkedDocument(
                            chunk_id=row["chunk_id"],
                            document_title=row["document_title"],
                            chunk_index=row["chunk_index"],
                            chunk_text=row["chunk_text"],
                            document_id=row["document_id"],
                            embedding=embedding,
                            metadata=json.loads(row["chunk_metadata"])
                            if isinstance(row["chunk_metadata"], str)
                            else {},
                        )
                        doc.chunks.append(chunk)

                    # Only add documents with valid chunks
                    if doc.chunks:
                        loaded_docs.append(doc)
                        total_chunks += len(doc.chunks)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        # Only proceed if we have documents to upsert
        if not loaded_docs:
            return

        # Create an async iterator from the loaded documents
        doc_iterator = list_to_async_iterator(loaded_docs)

        # Use the existing method to upsert all docs in this batch
        success, failed = await vector_store.upsert_document_chunks(doc_iterator)

        successful_docs += success
        failed_docs += failed

    # Process all files in batches
    file_batches = [
        parquet_files[i : i + batch_size]
        for i in range(0, len(parquet_files), batch_size)
    ]

    # Create progress bar with tqdm
    progress_bar = tqdm(total=len(file_batches), desc="Processing file batches")

    # Process batches concurrently
    tasks = []
    for batch in file_batches:
        task = asyncio.create_task(process_batch(batch))

        # Add callback to update progress bar when batch completes
        task.add_done_callback(lambda _: progress_bar.update(1))
        tasks.append(task)

    # Wait for all batches to complete
    await asyncio.gather(*tasks)
    progress_bar.close()

    print(
        f"Import complete! Successfully processed {successful_docs} documents, "
        f"failed: {failed_docs}, total chunks: {total_chunks}",
    )


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    objs = hydra.utils.instantiate(cfg)
    bm_instance = objs.bm
    vectoriser: ChromaDBEmbeddings = objs.vectoriser

    # Run the import with customized settings
    task = import_parquet_files_to_chroma(
        vectoriser,
        concurrency=20,  # Adjust based on your system capabilities
        batch_size=50,  # Adjust for optimal ChromaDB upsert size
    )
    asyncio.run(task)


if __name__ == "__main__":
    main()
