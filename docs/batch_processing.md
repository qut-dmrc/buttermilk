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

To create a new batch job, use the CLI with the batch config:

```bash
python -m buttermilk.runner.cli ui=batch flow=your_flow_name
```

This uses the default settings from `conf/run/batch.yaml`.

### Custom Configuration

You can customize batch parameters with Hydra overrides:

```bash
python -m buttermilk.runner.cli ui=batch flow=your_flow_name batch.concurrency=10 batch.max_records=100
```

### Available Commands

The batch CLI supports several commands:

- **create**: Create a new batch job (default)
- **list**: List all existing batch jobs
- **status**: Check status of a specific batch
- **cancel**: Cancel a running batch

Example:

```bash
# List all batches
python -m buttermilk.runner.cli ui=batch batch.command=list

# Check status of a specific batch
python -m buttermilk.runner.cli ui=batch batch.command=status batch.batch_id=batch-12345678

# Cancel a batch
python -m buttermilk.runner.cli ui=batch batch.command=cancel batch.batch_id=batch-12345678
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
    "flow": "my_flow",
    "shuffle": true,
    "max_records": 100,
    "concurrency": 5,
    "error_policy": "continue",
    "parameters": {"my_param": "value"}
  }'
```

Check batch status:

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/api/batches/batch-12345678' \
  -H 'accept: application/json'
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `flow` | The name of the flow to execute | (required) |
| `shuffle` | Whether to shuffle record IDs | `true` |
| `max_records` | Maximum number of records to process | `null` (all) |
| `concurrency` | Number of jobs to process concurrently | `5` |
| `error_policy` | How to handle job failures (`continue` or `stop`) | `continue` |
| `interactive` | Whether this batch requires interactive input | `false` |
| `parameters` | Additional parameters to pass to each flow run | `{}` |

## Interactive vs. Non-Interactive Mode

By default, batches run in non-interactive mode, but you can enable interactive mode if needed:

```bash
python -m buttermilk.runner.cli ui=batch flow=your_flow_name batch.interactive=true
```

In interactive mode, the system will handle user interaction points in the flow. This is useful for flows that require user input at certain steps.

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
