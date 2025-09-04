# Dynamic Configuration Reload System

This document describes the dynamic configuration reload system that allows updating Buttermilk flow configurations without restarting the server.

## Overview

The dynamic configuration reload system consists of:

1. **GCS-mounted configuration directory** - Configurations can be loaded from Google Cloud Storage
2. **Runtime configuration reload** - API endpoint to reload configurations without server restart
3. **Frontend admin interface** - Web UI for managing configuration reloads
4. **Configuration snapshots** - Automatic saving of configs used for each flow execution

## Architecture

### GCS Configuration Mounting

The Docker container can mount a GCS bucket to `/src/buttermilk/buttermilk/conf` using `gcsfuse`:

```bash
# Environment variables
export GCS_CONFIG_BUCKET="your-config-bucket"
export GCS_CONFIG_PATH="buttermilk-configs"  # Path within bucket

# The startup script will mount:
gcsfuse --implicit-dirs --only-dir="$GCS_CONFIG_PATH" "$GCS_CONFIG_BUCKET" /src/buttermilk/buttermilk/conf
```

### Configuration Reload API

Two new admin endpoints are available:

#### GET /api/admin/config-status
Returns current configuration status:
```json
{
  "flows_loaded": ["flow1", "flow2"],
  "flow_count": 2,
  "config_directory": "/src/buttermilk/buttermilk/conf",
  "config_exists": true,
  "config_timestamps": {"config.yaml": 1693747200},
  "gcs_bucket_env": "your-config-bucket",
  "timestamp": "2023-09-03T14:30:00Z"
}
```

#### POST /api/admin/reload-config
Triggers configuration reload:
```json
{
  "success": true,
  "flows_loaded": ["flow1", "flow2", "flow3"],
  "flows_updated": ["flow1", "flow2"],
  "flows_removed": [],
  "errors": [],
  "timestamp": "2023-09-03T14:30:00Z",
  "config_source": "/src/buttermilk/buttermilk/conf"
}
```

### Frontend Admin Panel

In the web interface sidebar, an "ADMIN" section provides:

- **Configuration Status**: Shows current mount status, flow count, GCS bucket info
- **Configuration Reload**: Button to trigger reload with real-time feedback
- **Reload History**: Shows results of the last reload attempt

### Configuration Snapshots

Every flow execution automatically saves a configuration snapshot to `/tmp/runs/{session_id}/config_snapshot/`:

```
/tmp/runs/session_123/config_snapshot/
├── latest.json                           # Latest snapshot metadata
├── flow_name_20230903_143000.json       # Timestamped flow config
└── flow_name_20230903_144500.json       # Additional snapshots
```

Snapshot contents include:
- Complete flow configuration used
- Runtime parameters and inputs
- Timestamp and session information
- All available flows at execution time

## Usage

### 1. Deploy with GCS Configuration

```dockerfile
# In your deployment
ENV GCS_CONFIG_BUCKET=my-buttermilk-configs
ENV GCS_CONFIG_PATH=production-configs

# The container will automatically mount the bucket
```

### 2. Update Configurations

Upload new/modified YAML files to your GCS bucket:
```bash
gsutil cp updated-flow.yaml gs://my-buttermilk-configs/production-configs/flows/
```

### 3. Reload in Production

Option A - Use the web interface:
1. Open Buttermilk web interface
2. Expand "ADMIN" section in sidebar
3. Click "RELOAD CONFIG"
4. Verify success in status display

Option B - Use the API directly:
```bash
curl -X POST http://your-server/api/admin/reload-config
```

### 4. Verify Configuration

Check that new flows are loaded:
```bash
curl http://your-server/api/admin/config-status
```

## Safety Features

### Error Handling
- Configuration validation before applying changes
- Rollback to previous configuration on reload failure
- Active sessions continue with original configuration
- Non-disruptive to running flows

### Reproducibility
- All flow executions save configuration snapshots
- Snapshots include exact configuration used
- Timestamped for audit trail
- Includes runtime parameters and inputs

### Monitoring
- Detailed logging of all reload operations
- Status tracking for GCS mount health
- Error reporting with specific failure details

## Troubleshooting

### GCS Mount Issues

Check mount status:
```bash
# In container
mount | grep gcsfuse
ls -la /src/buttermilk/buttermilk/conf/
```

Common issues:
- Service account permissions for GCS bucket
- Network connectivity to GCS
- Bucket or path doesn't exist

### Configuration Reload Failures

Check logs for specific errors:
```bash
# View FlowRunner logs
docker logs container_name | grep "Configuration reload"
```

Common issues:
- Invalid YAML syntax in configuration files
- Missing required configuration fields
- Hydra configuration inheritance problems

### Recovery

If configuration becomes corrupted:
1. Fix configuration files in GCS bucket
2. Use admin panel to reload
3. If reload fails, restart container to restore local configs
4. Check configuration snapshots for last known good config

## Best Practices

### Configuration Management
- Use version control for configuration files
- Test configurations in staging environment
- Use descriptive commit messages for config changes
- Monitor configuration reload success/failure

### Production Deployments
- Always verify configuration syntax before uploading
- Use staged rollouts for major configuration changes
- Keep backups of working configurations
- Monitor system behavior after configuration changes

### Performance Considerations
- Configuration reload is fast (< 1 second typically)
- Active sessions are not interrupted
- New flows use updated configuration immediately
- GCS mounting adds minimal overhead