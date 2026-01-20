# Groupchat Flow Debugging

Evaluate substantive success of groupchat flows. Exit code 0 proves nothing - follow this hierarchy.

## Validation Hierarchy

| Level | Check | What It Proves |
|-------|-------|----------------|
| 1 | Exit 0, no exceptions | Didn't crash |
| 2 | Input data correct | Flow received valid input |
| 3 | Config applied | Parameters match expectations |
| 4 | Structural completeness | All agents participated |
| 5 | Data quality | Outputs have required fields |
| 6 | Semantic validation | **Actually worked** |

## GCS Locations

Session data is saved to GCS under the session path:

```
gs://{bucket}/runs/{PROJECT}/batch/session-{timestamp}-{id}/
├── initial_bm_config_*.json    # Resolved hydra config
└── sessions/
    └── {groupchat_id}_*.json   # Session messages (now .jsonl)
```

### Find Session Path

From logs, search for "Successfully saved data":
```bash
grep "initial_bm_config" /tmp/bm_*.jsonl | jq -r '.event' | head -1
```

Or construct from execution context:
```
gs://prosocial-dev/runs/{PROJECT}/batch/session-{TIMESTAMP}-{ID}/
```

## Config Audit

**Verify config was applied correctly:**

```bash
# Get resolved config from GCS
gsutil cat "gs://prosocial-dev/runs/TJA/batch/session-{ID}/initial_bm_config_*.json" | jq '.cfg | {
  llms: .llms,
  limit: .run.limit,
  criteria: .run.pipeline.processors[0].variants.criteria
}'
```

**Key fields to verify:**
- `cfg.llms.judgers` - which models used
- `cfg.run.limit` - record limit applied
- `cfg.run.pipeline.processors[].variants` - parameter expansion

## Session Log Analysis

### Check Message Types (pre-JSONL fix)

```bash
gsutil cat "gs://.../sessions/*.json" | jq '[.messages[].type] | group_by(.) | map({type: .[0], count: length})'
```

**Expected in-band types** (post-JSONL fix):
- `AgentOutput` - agent processing results
- `ToolOutput` - tool/function results
- `UserResponseMessage` - user feedback
- `BaseRecord` subclasses - data records

**OOB types (control signals, now excluded):**
- `AgentAnnouncement`, `ConductorRequest`, `FlowEvent`
- `TaskProcessingStarted`, `TaskProcessingComplete`
- `StepRequest`, `FlowProgressUpdate`

### JSONL Session Logs (new format)

```bash
# Count messages by type
cat sessions/*.jsonl | jq -r '.type' | sort | uniq -c

# Check for errors in any message
cat sessions/*.jsonl | jq 'select(.content.error != null and .content.error != [])'

# Extract agent outputs
cat sessions/*.jsonl | jq 'select(.type == "AgentOutput") | .content'
```

## Application Log Analysis

```bash
# Quick summary
uv run python -m buttermilk.debug.ws_debug_cli analyze --file /tmp/bm_*.jsonl

# Check warnings
cat /tmp/bm_*.jsonl | jq -r 'select(.level == "warning") | [.timestamp, .event] | @tsv'

# Records processed
cat /tmp/bm_*.jsonl | jq -r 'select(.event | test("record|processed|completed"; "i")) | [.timestamp, .event] | @tsv' | head -10

# Agent completions
cat /tmp/bm_*.jsonl | jq -r 'select(.event | test("finished task"; "i")) | [.timestamp, .event] | @tsv'
```

## Common False Positives

| Claim | Why It's Insufficient |
|-------|----------------------|
| "Exit code 0" | Only means no crash |
| "47 tests passed" | Tests might use fakes/mocks |
| "Flow completed successfully" | Completion ≠ correctness |
| "No errors in output" | Errors might be in different fields |
| "Agent produced output" | Output might be garbage |
| "Session log exists" | May contain only OOB messages |

## Verification Checklist

Before declaring a flow "worked":

```
[ ] 1. Config audit: gsutil cat config → verify llms, limit, criteria
[ ] 2. Input validation: Record ID exists, not placeholder
[ ] 3. Record count: limit=N → exactly N processed
[ ] 4. Agent participation: All expected agents finished tasks
[ ] 5. No error fields: Check .error in outputs
[ ] 6. Output schema: Required fields present
[ ] 7. MANUAL: Read actual output content, verify semantic validity
```

## Workflow

1. **Find session**: `grep "initial_bm_config" /tmp/bm_*.jsonl`
2. **Audit config**: `gsutil cat gs://.../initial_bm_config_*.json | jq ...`
3. **Check logs**: `ws_debug_cli analyze --file /tmp/bm_*.jsonl`
4. **Read session**: `gsutil cat gs://.../sessions/*.jsonl | jq ...`
5. **Verify outputs**: Check actual agent output content
6. **Query traces**: (if needed) BigQuery for ExecutionTraces

## Related

- `docs/bots/DEBUGGING.md` - Full debugging reference
- `buttermilk/debug/CLAUDE.md` - Trace analysis tools
- `buttermilk/api/services/session_storage.py` - Session storage service (JSONL format)
