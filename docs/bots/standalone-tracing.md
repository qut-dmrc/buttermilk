# Standalone Tracing for Batch Processes

## Overview

When running Buttermilk agents outside of an orchestrator context (e.g., in batch processes, scripts, or CLI tools), agents need a parent trace context for proper tracing. This document describes the standalone tracing module that provides this functionality.

## Key Components

### 1. `StandaloneTraceContext` Class
Located in `buttermilk/_core/standalone_trace.py`, this class manages trace contexts for non-orchestrator workflows.

### 2. `create_standalone_trace` Context Manager
A convenience function that creates and manages a `StandaloneTraceContext` as an async context manager.

## Usage

### Basic Usage in Scripts

```python
from buttermilk._core.standalone_trace import create_standalone_trace

async def run_batch_process():
    async with create_standalone_trace("batch_job", job_type="vectorization") as trace:
        # Your processing code here
        # Agents will automatically use trace.trace_call as parent
        pass
```

### Passing to Agents

When agents are invoked within a standalone trace context, pass the trace call ID:

```python
agent_input = AgentInput(
    inputs={"data": data},
    parent_call_id=trace.get_call_id(),
)
result = await agent.invoke(agent_input)
```

### Integration with DocProcessor

The `DocProcessor` in `vector.py` now accepts an optional `parent_call` parameter that it passes to processors that support it (like the `Citator`).

## Implementation Details

1. **Agent Changes**: The `Agent` class now safely handles cases where `parent_call` is `None` (in `_core/agent.py` line 404).

2. **Citator Updates**: The `Citator` now accepts an optional `parent_call` parameter and passes it to agent invocations.

3. **Vector Pipeline**: The vector processing pipeline creates a standalone trace context for the entire run, providing proper tracing hierarchy.

## Benefits

- Proper trace hierarchy for batch processes
- Visibility into agent execution outside orchestrator contexts
- Consistent tracing across all Buttermilk workflows
- Easy debugging with Weave UI links