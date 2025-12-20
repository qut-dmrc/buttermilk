# Buttermilk Debug Module

Tools for debugging buttermilk flow executions.

## Debugging Execution Traces

When asked to debug a trace file (`/tmp/bm_*_exec-*.jsonl`), use the trace analysis tools:

```python
from pathlib import Path
from buttermilk.debug.trace_analysis import load_trace_file, get_errors, get_timeline

tf = load_trace_file(Path("/path/to/trace.jsonl"))
print(tf.summary)  # High-level overview

errors = get_errors(tf.traces)
for e in errors:
    print(f"{e.agent_name}: {e.error_message}")
```

## Available Functions

| Function | Purpose |
|----------|---------|
| `load_trace_file(path)` | Load trace with metadata, returns TraceFile |
| `summarize(traces)` | High-level summary (agents, errors, timing) |
| `get_errors(traces)` | All errors with context |
| `get_timeline(traces)` | Chronological execution view |
| `filter_traces(traces, **criteria)` | Filter by agent, time, error status |
| `get_trace(traces, call_id)` | Get specific trace by ID |
| `get_traces_by_agent(traces, name)` | All traces for one agent |
| `get_llm_conversation(trace)` | Extract LLM messages |
| `get_inputs_outputs(trace)` | Readable I/O format |

## CLI

```bash
uv run python -m buttermilk.debug trace /path/to/trace.jsonl --summary
uv run python -m buttermilk.debug trace /path/to/trace.jsonl --errors
uv run python -m buttermilk.debug trace /path/to/trace.jsonl --timeline
uv run python -m buttermilk.debug trace /path/to/trace.jsonl --agent scorer
```

## Common Workflows

**"What went wrong?"**: Load trace, check `tf.summary.error_count`, call `get_errors()`.

**"Why is output quality poor?"**: Filter to agent, inspect `get_inputs_outputs()` and `get_llm_conversation()`.

**"What was the execution path?"**: Check `tf.summary.execution_path` or use `get_timeline()`.
