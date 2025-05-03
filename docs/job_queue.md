# Job Queue System

This document explains the job queue system and how to use it.

## Overview

The Buttermilk job queue system uses Google Pub/Sub to manage batch processing jobs. The system consists of:

1. **Batch CLI** - Manages job creation and enqueuing
2. **Job Daemon** - Processes jobs from the queue when the system is idle
3. **FastAPI Integration** - Runs the job daemon alongside the web server

## Configuration

Configuration is stored in `conf/pubsub.yaml`. You'll need to set:

- `project` - Your Google Cloud project ID
- `jobs_topic` - Topic for job requests
- `status_topic` - Topic for status updates
- `subscription` - Subscription name
- `max_retries` - Maximum retry attempts for failed messages

## Running the Batch CLI

The batch CLI is used to create and enqueue jobs:

```bash
# Create a new batch job
python -m buttermilk.runner.batch_cli -c conf/config.yaml flow=your_flow_name

# List active batches
python -m buttermilk.runner.batch_cli -c conf/config.yaml run.batch.command=list

# Get status of a specific batch
python -m buttermilk.runner.batch_cli -c conf/config.yaml run.batch.command=status run.batch.batch_id=batch-12345678

# Cancel a batch
python -m buttermilk.runner.batch_cli -c conf/config.yaml run.batch.command=cancel run.batch.batch_id=batch-12345678
```

## Running the Job Daemon

The job daemon processes queued jobs when the system is idle. The daemon is automatically started when the FastAPI app runs. The FastAPI app tracks:
- API requests
- WebSocket connections 
- System activity
- When there's no user activity, it will process jobs from the queue.

## Architecture

### Job Flow

1. **Batch Creation**:
   - CLI creates a batch job with record IDs
   - Records are shuffled (if configured)
   - Jobs are published to the Pub/Sub queue

2. **Job Processing**:
   - Job daemon subscribes to the queue
   - When the system is idle, it pulls and processes jobs
   - Status updates are published back to the status topic
   
3. **Status Tracking**:
   - Job status (running, completed, failed) is tracked via Pub/Sub
   - CLI can query status of batches and individual jobs
