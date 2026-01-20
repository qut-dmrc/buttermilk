# Buttermilk Debugging Instructions

This file contains Buttermilk-specific debugging tools and workflows.

## Context Management for LLM Debugging

**CRITICAL for Claude/LLM agents**: Debugging generates large outputs that can overflow context.

| What | Location | How to use |
|------|----------|------------|
| **Console output** | Terminal | Monitor for errors/completion only - do NOT read full output into context |
| **Debug logs** | `/tmp/bm_*.jsonl` | NEVER read manually; use `analyze` and `logs -n` commands with small limits |
| **Session logs** | GCS: `gs://.../sessions/{groupchat_id}*.jsonl` | The substantive internal conversation - read THIS to understand how the groupchat worked |

**Rules**:
1. **Don't capture full flow output** - just run and watch for errors/completion
2. **Debug logs are verbose** - always use scripts (`analyze`, `logs -l ERROR`) not `cat`
3. **Session logs are the substance** - when you want to understand what actually happened in the groupchat, read the session log, not the debug log


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

- `project_name`: The project using buttermilk (e.g., `buttermilk`, `llm_reliability_study`)
- `execution_context_id`: Unique identifier like `exec-20251129T0615Z-YMqi-nicwin-nic`

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

## Post-Hoc Log Analysis

### Log Analysis Tips

**Note on Logs vs. Traces**:
- **Logs** (`/tmp/bm_*.jsonl`) contain structured application events (debug, info, error). Use `ws_debug_cli` to analyze these.
- **Execution Traces** are specialized objects (`ExecutionTrace`) often stored in BigQuery or exported separately. Do not attempt to use `trace_analysis` tools on standard log files.

**Problem**: Log outputs are too verbose for analysis. **Solution**:
### Step 1: Analyze (always start here)

```bash
# Get error summary - FIRST STEP for any debugging session
uv run python -m buttermilk.debug.ws_debug_cli analyze \
  --file /tmp/bm_llm_reliability_study_exec-20251129T0615Z-YMqi-nicwin-nic.jsonl

# Or analyze most recent log file
uv run python -m buttermilk.debug.ws_debug_cli analyze
```

Returns:
- Entry counts by severity level (debug, info, warning, error)
- Error counts by module
- Timeline (first/last timestamps)
- Sample error messages

### Step 2: Drill into specific logs

```bash
# View logs from any buttermilk JSONL file
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 \
  --file /tmp/bm_llm_reliability_study_exec-20251129T0615Z-YMqi-nicwin-nic.jsonl

# Filter to errors only
uv run python -m buttermilk.debug.ws_debug_cli logs -n 100 -l ERROR \
  --file /path/to/your/logfile.jsonl
```

### Direct JSONL analysis with jq

**Quick error summary**:

```bash
# Count entries by level
cat /tmp/your_log.jsonl | jq -r '.level' | sort | uniq -c

# List all error messages
cat /tmp/your_log.jsonl | jq -r 'select(.level=="error") | [.timestamp, .event] | @tsv'

# Show errors with context (module, function)
cat /tmp/your_log.jsonl | jq 'select(.level=="error") | {ts: .timestamp, event: .event, module: .module, func: .func_name}'
```

**Find exceptions and tracebacks**:

```bash
# Entries containing exception info
grep -i "exception\|traceback\|error" /tmp/your_log.jsonl | jq .

# Extract exception types
cat /tmp/your_log.jsonl | jq -r 'select(.exc_info != null) | .exc_info' | head -20
```

**Timeline analysis**:

```bash
# First and last timestamps
head -1 /tmp/your_log.jsonl | jq -r '.timestamp'
tail -1 /tmp/your_log.jsonl | jq -r '.timestamp'

# Events around a specific time (within 10 seconds of 06:15:30)
cat /tmp/your_log.jsonl | jq 'select(.timestamp | startswith("2025-11-29T06:15:3"))'
```

**Module-specific analysis**:

```bash
# List all modules that logged
cat /tmp/your_log.jsonl | jq -r '.module' | sort | uniq -c | sort -rn

# Show entries from a specific module
cat /tmp/your_log.jsonl | jq 'select(.module=="llms")'
```

### Key log fields

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 timestamp |
| `level` | Log level: debug, info, warning, error |
| `event` | Human-readable message |
| `module` | Python module name |
| `func_name` | Function that logged |
| `exc_info` | Exception traceback (if error) |
| `execution_context_id` | Links entries to same run |
| `project_name` | Project using buttermilk |

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

## Evaluating Substantive Flow Success

**Exit code 0 proves NOTHING** - it only means the process didn't crash. A flow can exit cleanly while producing garbage, skipping steps, or failing silently.

### Validation Hierarchy

**1. Basic Success** (necessary but NOT sufficient)
- Exit code 0
- No exceptions/tracebacks

**2. Input Validation** (often missed)
- Verify the flow received correct input data
- Check input was parsed properly
- If input is wrong, output validity is meaningless

**3. Config Audit**
- Verify actual config used matches your assumptions
- Check session log or hydra output for resolved config
- If `llms=cheap` wasn't actually applied, you won't know from output alone

**4. Structural Completeness**
- Expected number of records processed (with `limit=1`, exactly 1)
- Session log exists and is parseable (`.jsonl` in `sessions/`)
- All expected agents participated (check message types in session log)

**5. Data Quality**
- Read the session log - check actual `AgentOutput` and `ToolOutput` messages
- Look for `error` fields in outputs - empty means clean execution
- Verify outputs match expected schema (required fields, correct types)

**6. Semantic Validation** (requires human judgment)
- For judgment flows: Did the judge produce a score AND rationale?
- Is the output coherent and responsive to the input?
- Does the judgment contradict the input data?

### Session Log Analysis

The session log (`.jsonl` in `sessions/{groupchat_id}/`) captures in-band messages only:
- `ToolOutput` - tool/function results
- `AgentOutput` - agent processing results
- `UserResponseMessage` - user feedback
- `BaseRecord` subclasses - data records

**To validate a flow ran correctly:**

```bash
# 1. Find the session log
ls -la sessions/

# 2. Count messages by type
cat sessions/*.jsonl | jq -r '.type' | sort | uniq -c

# 3. Check for error fields in any message
cat sessions/*.jsonl | jq 'select(.content.error != null and .content.error != [])'

# 4. Extract agent outputs for review
cat sessions/*.jsonl | jq 'select(.type == "AgentOutput") | .content'
```

### What "Ran Perfectly" Actually Means

| Level | What it proves |
|-------|----------------|
| Exit 0 | Didn't crash |
| No errors in logs | No exceptions thrown |
| Session log exists | Messages were exchanged |
| Expected agent count | All agents participated |
| Output has required fields | Schema compliance |
| Output is semantically valid | **Actually worked** |

**The last level requires reading the actual output content and verifying it makes sense for the input.**

### Common False Positives

- ❌ "47 tests passed" - tests might use fakes/mocks
- ❌ "Flow completed successfully" - completion ≠ correctness
- ❌ "No errors in output" - errors might be in different fields or logs
- ❌ "Agent produced output" - output might be garbage

### Verification Checklist

Before declaring a flow "worked":

1. [ ] Config was applied as expected (check logs)
2. [ ] Input data was correct (not placeholder/empty)
3. [ ] Expected number of records processed
4. [ ] All expected agents produced output
5. [ ] No error fields in outputs
6. [ ] Output schema matches expectations
7. [ ] **Manual review**: Output content is semantically valid for input

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
