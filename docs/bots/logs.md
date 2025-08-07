# Buttermilk Log Analysis Guide

This guide provides comprehensive instructions for analyzing Buttermilk server logs.

## Finding the Current Debug Log

The current debug log is located at:
```bash
# Get the most recent debug log file
ls -t /tmp/buttermilk_*_debug.log 2>/dev/null | head -1

# Alternative: Use the helper script
./scripts/mcp_debug/getlog.sh
```

Debug logs are created when the server runs with `verbose=true` or in debug mode (`make debug`).

## Common Log Patterns and Searches

### 1. View Recent Log Entries
```bash
# View last N lines (replace 50 with desired number)
tail -50 /tmp/buttermilk_*_debug.log

# Follow log in real-time
tail -f /tmp/buttermilk_*_debug.log
```

### 2. Filter by Log Level
```bash
# Show only ERROR and CRITICAL messages
grep -E " - (ERROR|CRITICAL) - " /tmp/buttermilk_*_debug.log | tail -50

# Show WARNING and above
grep -E " - (WARNING|ERROR|CRITICAL) - " /tmp/buttermilk_*_debug.log | tail -50

# Show INFO and above
grep -E " - (INFO|WARNING|ERROR|CRITICAL) - " /tmp/buttermilk_*_debug.log | tail -50
```

### 3. Search for Errors and Exceptions
```bash
# Find errors, exceptions, and tracebacks
grep -iE "error|exception|traceback|failed" /tmp/buttermilk_*_debug.log | tail -50

# Find Python stack traces (multiline)
grep -A 10 "Traceback (most recent call last)" /tmp/buttermilk_*_debug.log
```

### 4. WebSocket and Message Flow
```bash
# WebSocket-related messages
grep -iE "websocket|ws:|message_service|flow.*message" /tmp/buttermilk_*_debug.log | tail -50

# Connection events
grep -iE "connect|disconnect|handshake" /tmp/buttermilk_*_debug.log | tail -50

# Message flow tracking
grep -iE "flow_id|message_id|sending.*message|received.*message" /tmp/buttermilk_*_debug.log | tail -50
```

### 5. API and HTTP Requests
```bash
# API endpoints hit
grep -E "POST|GET|PUT|DELETE|PATCH" /tmp/buttermilk_*_debug.log | tail -50

# Request/response details
grep -iE "request|response|status.*code|http" /tmp/buttermilk_*_debug.log | tail -50
```

### 6. Flow Execution
```bash
# Flow lifecycle events
grep -iE "flow.*start|flow.*complete|flow.*error|flow.*status" /tmp/buttermilk_*_debug.log | tail -50

# Agent interactions
grep -iE "agent|assistant|user.*message|tool.*call" /tmp/buttermilk_*_debug.log | tail -50
```

### 7. Performance and Timing
```bash
# Slow operations or timeouts
grep -iE "timeout|slow|performance|took.*seconds|duration" /tmp/buttermilk_*_debug.log | tail -50

# Token usage
grep -iE "token|usage|cost|model" /tmp/buttermilk_*_debug.log | tail -50
```

## Building Custom Grep Commands

### Basic Structure
```bash
grep [OPTIONS] "PATTERN" /tmp/buttermilk_*_debug.log | tail -N
```

### Useful Options
- `-i`: Case-insensitive search
- `-E`: Extended regex (use `|` for OR, `()` for grouping)
- `-A N`: Show N lines after match
- `-B N`: Show N lines before match
- `-C N`: Show N lines before and after match
- `-n`: Show line numbers
- `-c`: Count matches instead of showing them

### Complex Patterns
```bash
# Find specific flow errors
grep -E "flow_id.*12345.*error" /tmp/buttermilk_*_debug.log

# Find messages with specific user
grep -E "user_id.*abc123|user.*abc123" /tmp/buttermilk_*_debug.log | tail -50

# Find timeouts over 30 seconds
grep -E "timeout.*[3-9][0-9]|timeout.*[0-9]{3,}" /tmp/buttermilk_*_debug.log
```

## Time-based Filtering

```bash
# Get logs from last hour (approximate - checks timestamps)
grep "$(date -d '1 hour ago' '+%Y-%m-%d %H')" /tmp/buttermilk_*_debug.log

# Get logs from specific time range
awk '/2024-01-20 14:00/,/2024-01-20 15:00/' /tmp/buttermilk_*_debug.log
```

## Log File Management

```bash
# List all log files with details
ls -lht /tmp/buttermilk_*.log

# Get file size
du -h /tmp/buttermilk_*_debug.log

# Count total lines
wc -l /tmp/buttermilk_*_debug.log

# Find logs by date
find /tmp -name "buttermilk_*.log" -mtime -1  # Modified in last 24 hours
```

## Tips for Effective Log Analysis

1. **Start broad, then narrow**: Begin with general error searches, then focus on specific patterns
2. **Use context lines**: Add `-C 5` to see surrounding context for errors
3. **Combine filters**: Use pipes to chain grep commands for complex filtering
4. **Check timestamps**: Correlate errors with specific user actions or time periods
5. **Follow the flow**: Track flow_id or message_id through the logs to understand execution path

## Common Issues and Solutions

### No Debug Log Found
If no debug log exists:
1. Stop current server: `pkill -f buttermilk.runner.cli`
2. Start in debug mode: `make debug`
3. Or run with verbose: `uv run python -m buttermilk.runner.cli +run=api verbose=true`

### Log File Too Large
For very large logs:
```bash
# Use head/tail to limit scope
tail -100000 /tmp/buttermilk_*_debug.log | grep "pattern"

# Split search by time
grep "2024-01-20 14:" /tmp/buttermilk_*_debug.log | grep "error"
```

### Performance Considerations
- Use specific patterns to reduce matches
- Limit output with `tail` or `head`
- Consider using `ripgrep` (`rg`) for faster searches on very large files