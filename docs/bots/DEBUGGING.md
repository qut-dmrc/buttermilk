# Buttermilk Debugging Instructions

**Load framework DEBUGGING.md for generic debugging methodology** (`@$ACADEMICOPS/core/DEBUGGING.md`).

This file contains Buttermilk-specific debugging tools and workflows.

## Quick Reference

**Start debugging session**:

```bash
make debug                          # Start API server in background
uv run python -m buttermilk.debug.ws_debug_cli test-connection  # Verify
```

**View logs**:

```bash
uv run python -m buttermilk.debug.ws_debug_cli logs -n 20
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l ERROR
```

**Debug flow execution**:

```bash
# Start flow and capture messages
uv run python -m buttermilk.debug.ws_debug_cli start trans \
  --record betoota_snape_trans \
  --criteria tja

# Send manager response
uv run python -m buttermilk.debug.ws_debug_cli send "approved"

# Wait for completion
uv run python -m buttermilk.debug.ws_debug_cli wait --wait 30
```

**Stop debugging**:

```bash
make kill_api                       # Stop API server
```

## The ws_debug_cli Tool

**Buttermilk's primary debugging interface** for flow orchestration.

**All commands output JSON by default** (optimized for LLM agents). Use `--pretty` for human-readable output.

### Infrastructure Commands

**Test connection**:

```bash
uv run python -m buttermilk.debug.ws_debug_cli test-connection
# Expected: "Successfully connected to WebSocket"
```

**Check API health**:

```bash
curl -s http://localhost:8000/health
# Expected: {"status":"ok","message":"Core routes loaded"}
```

**List log files**:

```bash
uv run python -m buttermilk.debug.ws_debug_cli list-logs -n 5
```

**View logs**:

```bash
# Most recent bm_*.jsonl file
uv run python -m buttermilk.debug.ws_debug_cli logs -n 20

# Specific file
uv run python -m buttermilk.debug.ws_debug_cli logs --file /tmp/bm_project_session.jsonl

# Filter by level
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l ERROR
```

### Flow Control Commands

**Start flow** (returns JSON with session_id and messages):

```bash
uv run python -m buttermilk.debug.ws_debug_cli start trans \
  --record betoota_snape_trans \
  --criteria tja
```

**Send message** to current session:

```bash
uv run python -m buttermilk.debug.ws_debug_cli send "approved"

# Or to specific session
uv run python -m buttermilk.debug.ws_debug_cli send "approved" --session abc-123
```

**Wait for messages**:

```bash
# Wait 30 seconds, collect all messages
uv run python -m buttermilk.debug.ws_debug_cli wait --wait 30

# Filter by pattern
uv run python -m buttermilk.debug.ws_debug_cli wait --pattern "conclusion" --wait 10

# Filter by type
uv run python -m buttermilk.debug.ws_debug_cli wait --type ui_message --wait 10
```

**Session management**:

```bash
# Show current session
uv run python -m buttermilk.debug.ws_debug_cli session

# Clear saved session
uv run python -m buttermilk.debug.ws_debug_cli clear-session
```

## Valid System Parameters

**CRITICAL**: Flows require REAL data. Never use placeholder values.

**Available flows**: `trans`, and others from `conf/flows/`

**Valid criteria**: `tja`, `glaad`, and others from `conf/flows/criteria/`

**Record IDs MUST be real**:

- ✅ `betoota_snape_trans`, `kerri_colby_children_transitioning`
- ❌ `demo_record`, `test_record`, `placeholder`

Check your data sources for valid record IDs. **Flows WILL NOT run with arbitrary parameters.**

## Structured Logging

**Log format**: JSONL files in `/tmp/bm_*.jsonl`

**Log file naming**: `bm_{project_name}_{execution_context_id}.jsonl`

**Direct monitoring**:

```bash
tail -f /tmp/bm_*.jsonl
```

**Filtering logs**:

```bash
# Show errors
grep '"level":"ERROR"' /tmp/bm_*.jsonl | jq .

# Show specific component
grep '"logger":"buttermilk.flows"' /tmp/bm_*.jsonl | jq .

# Show timestamp and message
cat /tmp/bm_*.jsonl | jq -r '[.timestamp, .level, .event] | @tsv'
```

## DebugAgent Puppet Mode (Advanced)

**For programmatic flow control** in Python scripts or automated testing.

```python
from buttermilk.debug.ws_debug_cli import NonInteractiveDebugClient

client = NonInteractiveDebugClient(host="localhost", port=8000)

# Start flow and wait
result = await client.start_flow(
    flow_name="trans",
    query="Analyze this article",
    record="betoota_snape_trans",
    criteria="tja",
    wait_time=60,
)

print(f"Flow completed: {result['completed']}")
print(f"Messages: {len(result['messages'])}")
```

## Profiling Buttermilk Initialization

**Measure init performance**:

```bash
uv run python scripts/profile_init.py
```

**Analyze import times**:

```bash
python -X importtime -c "from buttermilk._core import config_bootstrap" 2> /tmp/import.txt
uv run python scripts/analyze_import_profile.py
```

**Find slow imports**:

```bash
cat /tmp/import.txt | grep "import time" | sort -k2 -rn | head -20
```

## Common Debugging Scenarios

**Flow hangs or doesn't complete**:

1. Check logs for errors: `ws_debug_cli logs -n 50 -l ERROR`
2. Verify all agents responding: Check message count increases
3. Check WebSocket connection: `test-connection`
4. Look for agent exceptions in logs

**Agent produces wrong output**:

1. Capture full message history: `ws_debug_cli start ... > flow_output.json`
2. Review agent inputs and outputs in messages
3. Check agent prompt and context
4. Verify data source provides expected input

**Configuration issues**:

1. Check structured logs at startup
2. Verify infrastructure initialized: Look for "Infrastructure setup complete"
3. Check env vars loaded: grep for "VERTEX_AI" in logs
4. Verify Hydra config resolution: Check for config errors in logs

**API connection failures**:

1. Check API health: `curl http://localhost:8000/health`
2. Verify server running: `ps aux | grep buttermilk`
3. Check firewall/port availability
4. Review server startup logs

## Output Conciseness

When using debugging tools, **extract key findings** instead of dumping full output:

- ✅ "Error in judge agent at line 45: NoneType has no attribute 'score'"
- ✅ "Flow completed in 23 messages, conclusion reached successfully"
- ❌ [Paste 200 lines of raw log output]

**Limit excerpts to 10-15 lines maximum per tool invocation.**

See framework DEBUGGING.md for generic profiling and systematic troubleshooting.

See `bots/docs/_CHUNKS/DEBUGGING.md` for complete ws_debug_cli command reference.
