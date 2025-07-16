# Buttermilk Debugging Commands

Custom Claude Code slash commands for debugging Buttermilk flows.

## Quick Start

1. **Check server health**: `/debug-health`
2. **Test connection**: `/debug-test` 
3. **Start flow**: `/debug-start zot`
4. **Send message**: `/debug-send "What are the principles of content moderation?"`
5. **Check logs**: `/debug-logs`

## Available Commands

### Basic Debugging
- **`/debug-health`** - Check API server health
- **`/debug-test`** - Test WebSocket connection  
- **`/debug-logs`** - Show recent error logs
- **`/debug-server <action>`** - Control server (start/stop/restart/status)

### Flow Control
- **`/debug-start <flow> [query]`** - Start a flow session
- **`/debug-send <message> [session_id]`** - Send message to active session
- **`/debug-monitor [session_id] [wait_seconds]`** - Monitor session activity

### Advanced Debugging
- **`/debug-demo [flow]`** - Complete debugging demonstration
- **`/debug-agents`** - Check agent tool registration

## Typical Debugging Workflow

```
/debug-health                           # 1. Check if server is running
/debug-test                            # 2. Test WebSocket connection
/debug-start zot                       # 3. Start flow with test query
/debug-send "Please provide citations" # 4. Send follow-up message
/debug-logs                           # 5. Check for any errors
```

## Troubleshooting

### Server Not Running
```
/debug-server start
/debug-health
```

### Flow Not Responding  
```
/debug-logs
/debug-agents
```

### Complete Reset
```
/debug-server restart
/debug-test
/debug-start zot
```

## Examples

### Content Moderation Research
```
/debug-start zot "What are the key principles of content moderation?"
/debug-send "Focus on academic research with citations"
/debug-monitor
```

### OSB Analysis
```
/debug-start osb "What are Facebook's content policies?"
/debug-send "Compare with Twitter's approach"
```

### Full Demonstration
```
/debug-demo zot
```

## Tips

- Use `--json-output` flag internally for structured responses
- Session IDs are automatically saved for follow-up commands
- Wait times are optimized for typical flow response patterns
- Commands work with all three flows: `zot`, `osb`, `trans`