# Asynchronous Batch Processing with a Job Queue

This document explains how Buttermilk can leverage a job queue system (e.g., Google Pub/Sub) for robust, asynchronous batch processing. This setup is ideal for large workloads or when you want to decouple job submission from job execution.

## Overview

When configured to use a job queue, Buttermilk's batch processing capabilities are extended:

1.  **Job Submission**: Users create batch jobs using the standard Buttermilk CLI (`ui=batch`) or API. If the system is configured for queuing, these batch requests are translated into messages and sent to a job queue.
2.  **Job Queue**: A message broker (like Google Pub/Sub) manages these messages (jobs). Each message might represent a single record to be processed or a bundle of records.
3.  **Worker Processes**: Dedicated Buttermilk worker processes (running in `ui=pub/sub` mode, as described in **[cli_commands.md](cli_commands.md)**) listen to this queue. They pick up jobs and execute the specified flow for each record.
4.  **Status Tracking**: Workers report status updates (e.g., to a database or another Pub/Sub topic), which can then be queried via the Buttermilk API or CLI to monitor batch progress.

This document primarily focuses on the Google Pub/Sub implementation.

## Configuration (Google Pub/Sub Example)

To use Google Pub/Sub as the job queue, you need to configure it in your Hydra settings, typically in a file like `conf/pubsub/default.yaml` or `conf/core/default.yaml` if it's a core component. Key settings include:

-   `project`: Your Google Cloud project ID.
-   `jobs_topic`: The Pub/Sub topic where new job requests (batches or individual records) are sent.
-   `status_topic`: A Pub/Sub topic where workers might publish status updates (optional, status might also be written directly to a database).
-   `subscription`: The Pub/Sub subscription name that workers will use to pull jobs from the `jobs_topic`.
-   `max_retries`: Maximum retry attempts for processing a message if a job fails. Behavior after max retries (e.g., sending to a dead-letter queue) should also be configured.

Consult your Buttermilk administrator or deployment guide for the exact configuration structure.

## Submitting Jobs to the Queue

You interact with the batch system as usual, using the CLI or API described in **[batch_processing.md](batch_processing.md)**.

```bash
# Example: Create a new batch job that will be sent to the Pub/Sub queue
python -m buttermilk.runner.cli ui=batch flow=your_processing_flow batch.parameters.input_dataset=large_dataset.csv
```

If Buttermilk is configured to use the job queue for batch processing:
*   The `ui=batch` command (or the corresponding API call) will package the batch request.
*   Instead of processing it immediately in the same process, it will publish the job(s) to the configured Pub/Sub `jobs_topic`.
*   The command will likely return quickly after successfully submitting the job(s) to the queue. You then monitor progress using status commands.

For details on `ui=batch` commands for creating, listing, checking status, and canceling batches, please refer to **[batch_processing.md](batch_processing.md)**. The underlying mechanism (immediate processing vs. queuing) depends on the Buttermilk system's configuration.

## Running Worker Processes (Job Daemon)

To process jobs from the queue, you need to run one or more Buttermilk worker instances. These are started using the `ui=pub/sub` mode:

```bash
python -m buttermilk.runner.cli ui=pub/sub
```
*   These workers subscribe to the specified Pub/Sub `subscription`.
*   They pull job messages from the queue and execute the corresponding Buttermilk flow.
*   Multiple workers can be run concurrently on different machines to scale processing capacity.
*   Refer to **[cli_commands.md](cli_commands.md)** for more about the `ui=pub/sub` mode.

Some deployments might configure the main FastAPI server (`ui=api` mode) to also process jobs from the queue when it's idle (i.e., has no active API requests or user interactions). This depends on the specific deployment strategy.

## Architecture and Job Flow

1.  **Batch Request**: A user submits a batch request via the CLI (`ui=batch`) or API.
2.  **Enqueueing**: The Buttermilk system (if configured for queuing) breaks down the batch into one or more job messages and publishes them to the `jobs_topic` in Google Pub/Sub. Each message contains information about the flow to run and the record(s) to process.
3.  **Job Consumption**: Dedicated Buttermilk workers (`ui=pub/sub` mode) are subscribed to the `jobs_topic` via a `subscription`. They pull messages from the queue.
4.  **Job Execution**: For each message, a worker executes the specified flow on the given record(s).
5.  **Status Updates**: During and after processing, workers update the job status. This might involve:
    *   Writing to a central database (e.g., BigQuery, PostgreSQL).
    *   Publishing status messages to a `status_topic` in Pub/Sub.
6.  **Monitoring**: Users can monitor the overall batch status and individual job results using the `ui=batch batch.command=status` CLI command or corresponding API endpoints, which query the stored status information.

This asynchronous architecture allows for scalable and resilient batch processing, as job submission, execution, and monitoring are decoupled.
