from buttermilk._core.storage_config import StorageConfig
from buttermilk.storage.file import FileStorage
from buttermilk._core.bm_init import BM

# Create a storage config for the GCS path
config = StorageConfig(
    type="gcs",
    path="gs://prosocial-public/osb/03_osb_fulltext_summaries.json",
    columns={
        "record_id": "id",
        "content": "full_text",
        "metadata": {
            "title": "title",
            "case_number": "case_number",
            "url": "url",
            "summary": "summary"
        }
    }
)

# Create a FileStorage instance
storage = FileStorage(config)

# Try to iterate over the records
print("Attempting to read records from GCS...")
count = 0
for record in storage:
    count += 1
    if count <= 3:  # Print details of first 3 records
        print(f"Record {count}:")
        print(f"  ID: {record.record_id}")
        print(f"  Content length: {len(record.content) if record.content else 0}")
        print(f"  Content: {record.content[:100]}...")  # Print first 100 chars of content
        print(f"  Title: {record.metadata.get('title', 'No title')}")
        print(f"  Has summary: {'Yes' if record.metadata.get('summary') else 'No'}")
        print()
    if count >= 10:
        break

print(f"Successfully read {count} records")