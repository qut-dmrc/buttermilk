# Batch Processing in Buttermilk

This document explains how to use Buttermilk's batch processing capabilities to run flows on multiple records independently.

## Overview

Batch processing allows you to execute a flow against multiple records without manual intervention. Batches are created as individual pub/sub tasks and are currently run on demand. 

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
```

### Available Commands

The batch CLI supports several commands, specified using the `batch.command` parameter:

- **create**: Create a new batch job (this is the default if no `batch.command` is specified).
- **run**: Fetch n jobs from pub/sub and run.

Example Scenarios:

```bash
# Create a new batch for 'data_processing_flow' 
python -m buttermilk.runner.cli ui=batch flow=data_processing_flow

# Pull a task and wait for it to complete
python -m buttermilk.runner.cli ui=batch

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
