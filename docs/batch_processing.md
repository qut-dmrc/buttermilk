# Batch Processing in Buttermilk

This document explains how to use Buttermilk's batch processing capabilities to run flows on multiple records independently.

## Overview

Batch processing allows you to execute a flow against multiple records without manual intervention. The system:

- Extracts record IDs from a flow's data sources
- Optionally shuffles them
- Processes each record independently with a fresh flow orchestrator instance
- Tracks progress and handles errors

## Usage Methods

You can use batch processing in three ways:

1. **Command Line Interface**: For manual batch creation and monitoring
2. **API Endpoints**: For programmatic batch management 
3. **Pub/Sub Integration**: For asynchronous batch processing (coming soon)

## Command Line Usage

### Creating a Batch

To create a new batch job, use the CLI with the batch config. The default command is `create`.

```bash
# Create a batch with default settings from conf/run/batch.yaml
python -m buttermilk.runner.cli ui=batch flow=your_flow_name
```

This uses the default settings found in `conf/run/batch.yaml`.

### Custom Configuration

You can customize batch parameters directly using Hydra overrides. This is useful for tailoring a specific batch run without altering your main configuration files.

```bash
# Create a batch with custom concurrency and max records
python -m buttermilk.runner.cli ui=batch flow=your_flow_name batch.concurrency=10 batch.max_records=100

# Create a batch and specify a particular dataset or input parameters for the flow
python -m buttermilk.runner.cli ui=batch flow=your_flow_name batch.parameters.input_file=data/my_records.csv batch.shuffle=false
```

### Available Commands

The batch CLI supports several commands, specified using the `batch.command` parameter:

- **create**: Create a new batch job (this is the default if no `batch.command` is specified).
- **list**: List all existing batch jobs.
- **status**: Check the status of a specific batch job using its ID.
- **cancel**: Cancel a running or pending batch job using its ID.

Example Scenarios:

```bash
# Create a new batch for 'data_processing_flow' and wait for it to complete
python -m buttermilk.runner.cli ui=batch flow=data_processing_flow batch.wait=true

# List all batches
python -m buttermilk.runner.cli ui=batch batch.command=list
# Example Output:
# batch-12345678  my_analysis_flow  RUNNING   2023-10-26T10:30:00Z
# batch-abcdef01  data_ingest_flow  COMPLETED 2023-10-25T15:00:00Z
# batch-7890ghjk  error_test_flow   FAILED    2023-10-26T11:00:00Z

# Check status of a specific batch
python -m buttermilk.runner.cli ui=batch batch.command=status batch.batch_id=batch-12345678
# Example Output:
# Batch ID: batch-12345678
# Flow: my_analysis_flow
# Status: RUNNING
# Submitted: 2023-10-26T10:30:00Z
# Started: 2023-10-26T10:30:05Z
# Progress: 75/100 records processed
# Concurrency: 5
# Error Policy: continue

# Cancel a batch
python -m buttermilk.runner.cli ui=batch batch.command=cancel batch.batch_id=batch-12345678
# Example Output:
# Batch batch-12345678 cancellation requested.
```

### Waiting for Completion

To create a batch and wait for it to complete:

```bash
python -m buttermilk.runner.cli ui=batch flow=your_flow_name batch.wait=true
```

## API Usage

The following API endpoints are available for batch management:

- `POST /api/batches`: Create a new batch job
- `GET /api/batches`: List all batches
- `GET /api/batches/{batch_id}`: Get status of a specific batch
- `DELETE /api/batches/{batch_id}`: Cancel a batch
- `GET /api/batches/{batch_id}/jobs`: Get all jobs for a batch
- `GET /api/batches/{batch_id}/jobs/{record_id}`: Get result of a specific job

### Example API Request

Create a new batch job:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/batches' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "flow_name": "my_analysis_flow",
    "shuffle": true,
    "max_records": 100,
    "concurrency": 5,
    "error_policy": "continue",
    "flow_parameters": {"input_source": "dataset_alpha.csv", "output_table": "results_alpha"}
  }'
# Example Response (Success 201 Created):
# {
#   "batch_id": "batch-newlycreated01",
#   "status": "PENDING",
#   "message": "Batch job created successfully.",
#   "details_url": "/api/batches/batch-newlycreated01"
# }
```

List all batches:
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/api/batches' \
  -H 'accept: application/json'
# Example Response (Success 200 OK):
# [
#   {
#     "batch_id": "batch-12345678",
#     "flow_name": "my_analysis_flow",
#     "status": "RUNNING",
#     "submitted_at": "2023-10-26T10:30:00Z",
#     "started_at": "2023-10-26T10:30:05Z",
#     "progress": {"processed": 75, "total": 100, "errors": 2},
#     "details_url": "/api/batches/batch-12345678"
#   },
#   {
#     "batch_id": "batch-abcdef01",
#     "flow_name": "data_ingest_flow",
#     "status": "COMPLETED",
#     "submitted_at": "2023-10-25T15:00:00Z",
#     "started_at": "2023-10-25T15:00:05Z",
#     "completed_at": "2023-10-25T16:30:00Z",
#     "progress": {"processed": 500, "total": 500, "errors": 0},
#     "details_url": "/api/batches/batch-abcdef01"
#   }
# ]
```

Check batch status:

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/api/batches/batch-12345678' \
  -H 'accept: application/json'
# Example Response (Success 200 OK):
# {
#   "batch_id": "batch-12345678",
#   "flow_name": "my_analysis_flow",
#   "status": "RUNNING",
#   "shuffle": true,
#   "max_records": 100,
#   "concurrency": 5,
#   "error_policy": "continue",
#   "flow_parameters": {"input_source": "dataset_alpha.csv", "output_table": "results_alpha"},
#   "submitted_at": "2023-10-26T10:30:00Z",
#   "started_at": "2023-10-26T10:30:05Z",
#   "updated_at": "2023-10-26T11:00:15Z",
#   "progress": {
#     "total_records_in_source": 250, # Example: if flow identified more records than max_records
#     "records_to_process": 100,
#     "records_processed": 75,
#     "records_succeeded": 73,
#     "records_failed": 2,
#     "estimated_time_remaining": "approx. 15 minutes"
#   },
#   "jobs_url": "/api/batches/batch-12345678/jobs"
# }
```

## Configuration Options

| Option (`batch.<option_name>`) | API Field (`request_body.<field_name>`) | Description | Default |
|--------------------------------|-----------------------------------------|-------------|---------|
| `flow`                         | `flow_name`                             | The name of the flow to execute (must correspond to a flow definition known to Buttermilk). | (required) |
| `shuffle`                      | `shuffle`                               | Whether to shuffle record IDs obtained from the flow's data sources before processing. | `true` |
| `max_records`                  | `max_records`                           | Maximum number of records to process from the data source. If `null` or not provided, all records will be processed. | `null` (all) |
| `concurrency`                  | `concurrency`                           | Number of jobs (records) to process concurrently. Adjust based on system resources and external API rate limits. | `5` |
| `error_policy`                 | `error_policy`                          | How to handle job failures: `continue` (process other records) or `stop` (halt the entire batch). | `continue` |
| `interactive`                  | `interactive`                           | Whether this batch requires interactive input. If `true`, the flow may pause for user input. | `false` |
| `parameters`                   | `flow_parameters`                       | A dictionary of additional parameters to pass to each individual flow run initiated by the batch. These can override flow defaults or provide runtime specifics. | `{}` |
| `wait`                         | (N/A for API - client polls)            | (CLI only) If `true`, the CLI command will block until the batch completes or fails. | `false` |
| `batch_id`                     | (URL parameter for GET/DELETE)          | (CLI only for status/cancel) The unique identifier of the batch. | (N/A for create) |
| `command`                      | (N/A for API - HTTP methods used)       | (CLI only) The operation to perform: `create`, `list`, `status`, `cancel`. | `create` |

## Interactive vs. Non-Interactive Mode

By default, batches run in non-interactive mode. This means the flow execution for each record is expected to run to completion without requiring real-time user input.

You can enable interactive mode if your flow is designed to include points where user interaction is necessary:
```bash
python -m buttermilk.runner.cli ui=batch flow=your_flow_name batch.interactive=true
```

In interactive mode, the system will attempt to handle user interaction points defined within the flow. This is useful for scenarios like human-in-the-loop validation, where a person needs to confirm or provide input for certain records, even when processing a larger batch. Be mindful that interactive batches may require significantly more time to complete if user input is frequently required.

## Architecture

The batch processing system consists of:

1. **BatchRunner**: Manages batch jobs and tracks progress
2. **API Endpoints**: For programmatic control
3. **CLI Interface**: For command-line usage
4. **Configuration**: Hydra-based configuration

The implementation follows a daemon-oriented architecture where the API server acts as the central hub, with workers that pull jobs from a queue for execution.

## Best Practices

1. **Set Concurrency Carefully**: Higher concurrency uses more resources but processes faster
2. **Use Shuffle for Random Sampling**: Helps prevent processing bias
3. **Implement Proper Error Handling**: Use the appropriate error policy
4. **Monitor Long-Running Batches**: Either with the CLI status command or API endpoint
