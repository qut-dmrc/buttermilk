Next, let's create a bigquery record dataloader if we don't have one already. 
Core functionality:
1. Defaults to a generic `Records` table. This table has to be clustered by record_id and the dataset name. Its other fields should mirror the Record object fields.
2. Seamless integration with other dataloader objects
3. YAML file configuration -- start by converting the OSB and tox datasets from JSONL in GCS to records in separate bigquery tables.

Nice to have:
- streaming (batches of?) randomised Record rows
- additional cluster for split type ('train', 'test', etc?)
- A well thought out dataset import system, which provides a VERY easy to use interface for HASS scholars to map their dataset to the fields we expect for a Record and uploads the entire dataset to BigQuery (JSONL to GCS first, retaining the JSONL files)
- A well thought out record import method, integrated with the existing fetch agent preferably, providing a way within a workflow to fetch new individual records from a URL and, after converting to Record format, saving that record to GCS and BigQuery. Make it easy to extend so that users can write simple converters for a wide range of input sources (e.g. extract TikTok videos by url, news articles by URL, and Amazon pricing information and displayed recommendations for a given object page.)

---
Summary for Next Developer

  Context: Completed comprehensive backend update to prioritize native Buttermilk base classes throughout
  the API and frontend, eliminating unnecessary object conversions.

  ✅ Completed Work:

  1. API Simplification (DONE)
    - Removed all conversion layers from API routes
    - APIs now send native Record and AgentTrace objects directly using model_dump()
    - Fixed JSON serialization error that was causing 500s
  2. Frontend Native Object Support (DONE)
    - Updated ToxicityScoreTable.svelte to process AgentTrace objects directly
    - Updated ScoreMessagesDisplay.svelte to handle native AgentTrace arrays
    - Updated main score page to work with native objects
    - Removed all backwards compatibility code
  3. BigQuery Record Dataloader (DONE)
    - Created BigQueryRecordLoader with clustering by record_id and dataset_name
    - Added migration utility (migrate_to_bigquery.py) for JSONL → BigQuery conversion
    - Created YAML configs for osb and tox datasets
    - Integrated with existing dataloader factory

  🔄 Ready for Next Steps (User Priority: 3,2,1,4):

  1. #3: Fetch Agent Integration - Create record import method integrated with existing fetch agent for
  URL-based record creation with auto-save to GCS+BigQuery
  2. #2: Test API Endpoints - Verify native object serialization works correctly in practice
  3. #1: Fix Linting Issues - Clean up unused imports and whitespace issues
  4. #4: Streaming Enhancements - Add batching, connection pooling, progress callbacks

  🧰 Key Files Modified:

  - buttermilk/api/services/data_service.py - Returns native objects
  - buttermilk/api/routes.py - Uses model_dump() for serialization
  - buttermilk/data/bigquery_loader.py - New BigQuery loader
  - buttermilk/tools/migrate_to_bigquery.py - Migration utilities
  - Frontend components in buttermilk/frontend/chat/src/lib/components/score/

  📊 Cost Efficiency:

  - Total cost: $4.30
  - Code changes: +1052/-437 lines
  - Duration: 25m API time, 1h 51m wall time

  The architecture now follows the principle of using fundamental base classes throughout, enabling better
  extensibility and eliminating conversion overhead.