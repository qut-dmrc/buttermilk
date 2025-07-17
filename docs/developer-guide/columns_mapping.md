# Column Mapping in Buttermilk Data Loaders

## Overview

All Buttermilk data loaders support column mapping via the `columns` field in storage configurations. This allows you to rename fields from your data source to match the expected Record field names.

## Purpose

The `columns` field serves as a mapping dictionary to transform data source field names into Record field names. This is essential when:

- Your data source uses different field names than what Record expects
- You want to standardize field names across different data sources
- You need to map specific data fields to Record metadata

## Configuration

### Field Definition
```yaml
columns: {}  # Can be empty - no field renaming
# OR
columns:
  content: "text"           # Maps source "text" field to Record "content" field
  ground_truth: "expected"  # Maps source "expected" field to Record "ground_truth" field
  title: "name"            # Maps source "name" field to Record metadata "title"
```

### Dictionary Structure
- **Keys**: Target field names (what you want the field to be called in the Record)
- **Values**: Source field names (what the field is called in the original data)

Pattern: `{target_record_field: source_data_field}`

## Examples

### Basic Column Mapping
```yaml
# Example: JSONL file with different field names
storage:
  my_dataset:
    type: file
    path: gs://bucket/data.jsonl
    columns:
      content: "message_text"     # Source has "message_text", Record needs "content"
      record_id: "id"            # Source has "id", Record needs "record_id"
      title: "subject_line"      # Source has "subject_line", map to metadata "title"
```

### Multiple Field Mappings
```yaml
storage:
  survey_data:
    type: file  
    path: data/survey.csv
    columns:
      content: "response_text"
      ground_truth: "expert_rating"
      record_id: "survey_id" 
      title: "question_text"
      category: "survey_type"
```

### Empty Columns (No Mapping)
```yaml
storage:
  clean_data:
    type: file
    path: data/records.jsonl
    columns: {}  # No field renaming needed - source fields match Record fields
```

### BigQuery Column Mapping
```yaml
storage:
  legacy_table:
    type: bigquery
    project_id: my-project
    dataset_id: research
    table_id: legacy_content
    columns:
      content: "message_body"        # BigQuery column "message_body" → Record "content"
      record_id: "unique_id"         # BigQuery column "unique_id" → Record "record_id"
      ground_truth: "expert_score"   # BigQuery column "expert_score" → Record "ground_truth"
      title: "message_subject"       # BigQuery column "message_subject" → metadata "title"
```

## Supported Data Loaders

### ✅ Loaders with Column Mapping Support

1. **JSONLDataLoader** - For .jsonl files
2. **CSVDataLoader** - For .csv files  
3. **HuggingFaceDataLoader** - For HuggingFace datasets
4. **FileStorage** - Unified file storage system
5. **PlaintextDataLoader** - For text files (maps filename, content, path, size)
6. **BigQueryStorage** - For BigQuery tables with different column names

### ❌ Loaders without Column Mapping

1. **BigQueryRecordLoader** - Deprecated compatibility wrapper

## Record Field Targets

When mapping columns, you can target these Record fields:

### Core Record Fields
- `record_id` - Unique identifier for the record
- `content` - Main text/data content  
- `uri` - Resource identifier/URL
- `title` - Record title/name

### Metadata Fields
Any field not in the core Record fields will be automatically placed in the `metadata` dictionary.

Example:
```yaml
columns:
  content: "text"          # → record.content
  record_id: "id"         # → record.record_id  
  category: "type"        # → record.metadata["category"]
  confidence: "score"     # → record.metadata["confidence"]
```

## Implementation Details

### Processing Order
1. Data loader reads raw data from source
2. If `columns` mapping exists, apply field renaming
3. Separate core Record fields from metadata fields
4. Create Record object with mapped fields

### Default Behavior
- If `columns` is empty (`{}`), no field mapping is applied
- If `columns` is not specified, defaults to empty dictionary
- Original source data structure is preserved when no mapping exists

### Error Handling
- If a source field specified in `columns` doesn't exist, it's silently skipped
- If mapping targets an invalid Record field, it goes to metadata
- Loaders continue processing even if some column mappings fail

## Best Practices

### 1. Always Map Core Fields
Ensure your data source provides or maps to these essential fields:
```yaml
columns:
  record_id: "id"          # Required - unique identifier
  content: "text"          # Required - main content
```

### 2. Use Descriptive Source Names
Be explicit about which source fields you're mapping:
```yaml
columns:
  content: "article_full_text"      # Clear source field name
  title: "article_headline"        # Descriptive mapping
  ground_truth: "expert_label"     # Obvious purpose
```

### 3. Document Your Mappings
Add comments to explain non-obvious mappings:
```yaml
columns:
  content: "body"                  # Main article text
  ground_truth: "is_toxic"        # Binary toxicity label from experts
  title: "headline"               # Article title for display
```

### 4. Test with Sample Data
Verify your column mappings work with sample data before processing large datasets.

## Troubleshooting

### Common Issues

1. **Missing Required Fields**: Ensure `record_id` and `content` are mapped or exist in source
2. **Wrong Field Names**: Check source data field names match your `columns` values exactly
3. **Case Sensitivity**: Field names are case-sensitive in mapping
4. **Empty Results**: Verify source data contains the fields specified in `columns`

### Debugging
Enable debug logging to see column mapping in action:
```python
import logging
logging.getLogger("buttermilk.data.loaders").setLevel(logging.DEBUG)
```

## Related Documentation

- [Data Loaders Guide](data_loaders.md)
- [Storage Configuration](storage_config.md) 
- [Record Structure](records.md)