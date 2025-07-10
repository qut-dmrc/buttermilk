# Buttermilk Debugging Guide

## Overview

This guide explains how to effectively debug Buttermilk applications, with a focus on automated debugging capabilities that allow tools and scripts to observe system behavior.

## Debug Mode

### Starting the Server in Debug Mode

The recommended way to debug Buttermilk is using the debug mode which captures all output to a file:

```bash
# Start server with output capture
make debug

# The output is saved to: /tmp/buttermilk-debug.log
# You can tail it in real-time:
tail -f /tmp/buttermilk-debug.log
```

### Benefits of Debug Mode

1. **Persistent Logs**: All stdout and stderr are captured to `/tmp/buttermilk-debug.log`
2. **Tool Accessibility**: Automated tools can read the log file
3. **Real-time Monitoring**: Use `tail -f` to watch logs as they happen
4. **Complete Output**: Captures everything including print statements

## Debugging Workflow

### 1. Start Server in Debug Mode

```bash
# Stop any existing server
pkill -f buttermilk.runner.cli

# Start in debug mode
make debug &

# Wait for server to be ready
sleep 10

# Verify it's running
curl -s http://localhost:8000/health | jq
```

### 2. Run Your Test Case

```bash
# Example: Test with CLI
cd buttermilk/frontend/cli
node dist/cli.js --host localhost --port 8000

# Or run automated tests
./scripts/automated-flow-test.sh
```

### 3. Analyze the Logs

Use the provided log viewer script:

```bash
./scripts/view-logs.sh
```

Or manually search the logs:

```bash
# View recent errors
grep -i "error\|exception" /tmp/buttermilk-debug.log | tail -20

# View WebSocket messages
grep -i "websocket\|message_service" /tmp/buttermilk-debug.log | tail -50

# Search for specific patterns
grep -i "your_search_term" /tmp/buttermilk-debug.log
```

## Common Debugging Scenarios

### 1. WebSocket Message Issues

When debugging WebSocket communication problems:

```bash
# Look for message processing
grep "process_message_from_ui\|Unknown message type" /tmp/buttermilk-debug.log

# See all received WebSocket data
grep "Received data from WebSocket" /tmp/buttermilk-debug.log
```

Example output:
```
[MONITOR_UI] Received data from WebSocket: {'type': 'user_message', 'payload': {'text': 'hello'}}
WARNING message_service.py:204 Unknown message type received on websocket: user_message
```

### 2. Flow Execution Issues

To debug flow problems:

```bash
# Check flow initialization
grep -i "flow.*started\|flow.*failed" /tmp/buttermilk-debug.log

# View flow metrics
curl -s http://localhost:8000/monitoring/metrics/flows | jq

# Check for configuration issues
grep -i "error.*config\|missing.*required" /tmp/buttermilk-debug.log
```

### 3. Validation Errors

Pydantic validation errors are clearly shown:

```bash
# Find validation errors
grep -A5 "validation error" /tmp/buttermilk-debug.log
```

Example:
```
ERROR message_service.py:208 Error processing message: 3 validation errors for UIMessage
content
  Field required [type=missing, input_value={'type': 'ui_message', ...}, input_type=dict]
```

## Debug Information Available

### Server Logs Include

1. **Startup Information**
   - Configuration details
   - Cloud authentication status
   - API endpoints loaded

2. **Request Processing**
   - HTTP requests and responses
   - WebSocket connections and messages
   - Flow execution status

3. **Errors and Warnings**
   - Validation errors with details
   - Exception tracebacks
   - Configuration warnings

4. **Performance Metrics**
   - Memory usage warnings
   - Response times
   - System health status

## Automated Debugging

### For CI/CD and Testing Tools

The debug log file enables automated debugging:

```python
# Python example
def check_for_errors():
    with open('/tmp/buttermilk-debug.log', 'r') as f:
        logs = f.read()
        if 'ERROR' in logs:
            # Extract and report errors
            errors = [line for line in logs.split('\n') if 'ERROR' in line]
            return errors
    return []

# Check if a message was processed
def verify_message_processed(message_type):
    with open('/tmp/buttermilk-debug.log', 'r') as f:
        logs = f.read()
        return f"Unknown message type received on websocket: {message_type}" not in logs
```

### Integration with Monitoring

The debug log can be:
- Parsed by log aggregation tools
- Monitored for specific patterns
- Used to generate alerts
- Analyzed for performance metrics

## Best Practices

1. **Always Use Debug Mode for Development**
   ```bash
   make debug  # Instead of make api
   ```

2. **Clear Logs Between Test Runs**
   ```bash
   > /tmp/buttermilk-debug.log  # Clear the log
   make debug &
   ```

3. **Use Structured Searches**
   ```bash
   # Create a debug session
   DEBUG_SESSION=$(date +%Y%m%d_%H%M%S)
   echo "=== Debug Session $DEBUG_SESSION ===" >> /tmp/buttermilk-debug.log
   ```

4. **Monitor Specific Components**
   ```bash
   # Watch only WebSocket activity
   tail -f /tmp/buttermilk-debug.log | grep -i websocket
   ```

## Troubleshooting

### Log File Too Large

```bash
# Rotate the log
mv /tmp/buttermilk-debug.log /tmp/buttermilk-debug.log.old
> /tmp/buttermilk-debug.log
```

### Can't See Print Statements

Make sure you're using the logger instead of print:
```python
# Good
logger.info(f"Processing message: {message}")

# Also captured but less structured
print(f"[DEBUG] Processing message: {message}")
```

### Missing Log Output

1. Ensure server started with `make debug` not `make api`
2. Check file permissions: `ls -la /tmp/buttermilk-debug.log`
3. Verify tee is working: `ps aux | grep tee`

## Related Documentation

- [Automated Testing Guide](../AUTOMATED-TESTING-GUIDE.md) - How to use debug mode in tests
- [CLI Development Guide](buttermilk/frontend/cli/DEVELOPMENT.md) - Debugging CLI issues