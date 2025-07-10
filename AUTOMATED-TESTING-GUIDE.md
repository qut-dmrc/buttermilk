# Buttermilk Automated Testing Guide

## Overview

This guide documents what can be automated in the Buttermilk testing process and what requires manual intervention.

## What I Can Do Automatically

### 1. Server Management
- ✅ Start the Buttermilk backend server using `make api`
- ✅ Check if the server is running (health checks)
- ✅ Monitor server status and API endpoints
- ✅ Track flow execution metrics

### 2. CLI Client Testing
- ✅ Build the CLI client
- ✅ Run automated tests against real servers
- ✅ Execute flow requests via WebSocket
- ✅ Capture and analyze responses
- ✅ Generate test reports

### 3. Debugging
- ✅ Check API endpoints and schemas
- ✅ Monitor flow execution metrics
- ✅ Identify failing flows
- ✅ Test WebSocket connections
- ✅ Create debugging scripts

### 4. Monitoring
- ✅ Check system health
- ✅ View active alerts
- ✅ Track performance metrics
- ✅ Monitor WebSocket connections

## What Requires Manual Intervention

### 1. Configuration Issues
- ❌ **GCP Authentication**: The server requires GCP credentials and access to secrets
- ❌ **ChromaDB Setup**: The zot flow needs ChromaDB configured for vector search
- ❌ **LLM Configuration**: Requires access to `models.json` secret in GCP
- ❌ **Environment Variables**: Some configs may need manual setup

### 2. Server Logs
- ❌ **Log Access**: Server logs are sent to GCP Cloud Logging (not local files)
- ❌ **Error Details**: Detailed error messages require checking GCP logs
- ❌ **Debug Mode**: May need to restart server with debug flags

### 3. Flow-Specific Issues
- ❌ **Zot Flow**: Currently failing - likely needs ChromaDB configuration
- ❌ **Model Access**: Flows may fail if LLM models aren't accessible
- ❌ **Data Sources**: Some flows may need specific data sources configured

## Automated Testing Process

### Quick Test Script
```bash
# Run the automated test
./automated-flow-test.sh
```

This script will:
1. Check/start the server
2. Build the CLI
3. Test API connection
4. Run flow tests
5. Report metrics

### Manual Debugging Steps

When flows fail (like zot is currently):

1. **Check GCP Logs**:
   ```bash
   gcloud logging read "resource.type=cloud_run_job AND jsonPayload.flow=zot" --limit 50
   ```

2. **Verify Configuration**:
   - Check if ChromaDB is running
   - Verify GCP credentials are valid
   - Ensure models.json secret is accessible

3. **Test Individual Components**:
   ```bash
   # Test if vector store is working
   curl -X POST http://localhost:8000/mcp/agents/vector-query
   ```

## Current Status

### Working ✅
- Server starts and responds to health checks
- WebSocket connections establish successfully
- API endpoints are accessible
- Flow metrics are tracked

### Not Working ❌
- **Zot flow execution**: Fails immediately (2 failures, 0 successes)
- **Flow message delivery**: WebSocket connects but no flow messages received
- **Server log access**: Logs go to GCP, not local files

## Recommendations

1. **For Automated Testing**:
   - Use the provided test scripts
   - Monitor metrics endpoints
   - Check health endpoints regularly

2. **For Manual Debugging**:
   - Access GCP Cloud Logging for detailed errors
   - Verify all dependencies are configured
   - Check that required services (ChromaDB, etc.) are running

3. **For Development**:
   - Consider adding local logging option for development
   - Add more detailed error messages to flow failures
   - Implement better WebSocket error reporting

## Next Steps

To get the zot flow working:
1. Check GCP logs for the specific error
2. Verify ChromaDB is configured and accessible
3. Ensure the zot flow has required data/models
4. Test with a simpler flow (like 'test') first

The automated testing framework is ready and functional - the issue is with the flow configuration, not the testing infrastructure.