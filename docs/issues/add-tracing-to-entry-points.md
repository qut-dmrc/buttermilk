# Add Standalone Tracing to All Non-Orchestrator Entry Points

## Overview

Following the implementation of standalone tracing support (PR that adds `standalone_trace.py`), we need to extend tracing coverage to all entry points that run outside of an orchestrator context. This will ensure consistent observability across all execution modes.

## Background

The standalone tracing module (`buttermilk/_core/standalone_trace.py`) was created to provide parent trace contexts for agents running outside orchestrators. Currently implemented in:
- `buttermilk/data/vector.py` - Vector database batch processing

## Entry Points Requiring Tracing

### 1. CLI Modes (`buttermilk/runner/cli.py`)

#### Batch Mode (lines 98-100)
```python
case "batch":
    asyncio.run(flow_runner.create_batch(...))
```
- Creates batch jobs by adding RunRequests to queue
- Needs trace context for job creation process

#### Batch Run Mode (lines 102-112)
```python
case "batch_run":
    asyncio.run(flow_runner.run_batch_job(...))
```
- Processes jobs from queue
- Each job should have its own trace context

#### Pub/Sub Mode (lines 167-184)
```python
case "pub/sub":
    batch_cli_main(conf)
```
- Listens to Google Cloud Pub/Sub
- Each message processing should be traced

#### Console Mode (lines 75-96)
```python
case "console":
    asyncio.run(flow_runner.run_flow(...))
```
- Single flow execution
- Should have trace context for the entire run

### 2. API Endpoints (`buttermilk/api/flow.py`)

#### HTTP Request Handlers
- Session creation endpoints
- Flow execution via API calls
- Each request should create a trace context

#### WebSocket Connections (line 181)
```python
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(...)
```
- Long-lived connections need special handling
- Consider per-message or per-operation tracing

### 3. Slack Bot (`buttermilk/runner/slackbot.py`)

#### Event Handlers
```python
async def register_handlers(...)
```
- Each Slack command/event should create a trace
- Trace should span the entire flow execution

### 4. Streamlit Interface (line 114-127)
- Web UI interactions
- Each user action triggering a flow should be traced

### 5. Scheduled/Background Tasks
- Any cron jobs or periodic tasks
- Background cleanup or maintenance operations

## Implementation Approach

### 1. Create Tracing Decorators

```python
# buttermilk/_core/tracing_decorators.py
from functools import wraps
from buttermilk._core.standalone_trace import create_standalone_trace

def traced_entry_point(name: str, **default_attributes):
    """Decorator for adding tracing to entry points."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract relevant attributes from args/kwargs
            attributes = {**default_attributes}
            # Add dynamic attributes based on function args
            
            async with create_standalone_trace(name, **attributes) as trace:
                # Inject trace context if needed
                return await func(*args, trace_context=trace, **kwargs)
        return wrapper
    return decorator
```

### 2. Update Entry Points

Example for batch mode:
```python
async def run_batch_job(self, callback_to_ui: Callable, max_jobs: int = 1, 
                       wait_for_completion: bool = True, trace_context=None) -> None:
    """Pull and run jobs from the queue."""
    # If no trace context provided, create one
    if not trace_context:
        async with create_standalone_trace("batch_job_runner", max_jobs=max_jobs) as trace:
            await self._run_batch_job_impl(callback_to_ui, max_jobs, wait_for_completion, trace)
    else:
        await self._run_batch_job_impl(callback_to_ui, max_jobs, wait_for_completion, trace_context)
```

### 3. FastAPI Middleware

```python
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    """Add trace context to all HTTP requests."""
    attributes = {
        "method": request.method,
        "path": request.url.path,
        "client": request.client.host if request.client else None,
    }
    
    async with create_standalone_trace(f"http_{request.method}_{request.url.path}", **attributes) as trace:
        # Store trace in request state for handlers to use
        request.state.trace_context = trace
        response = await call_next(request)
        return response
```

## Testing Requirements

1. **Unit Tests**
   - Test trace context creation for each entry point
   - Verify trace attributes are correctly set
   - Test error handling within trace contexts

2. **Integration Tests**
   - Verify traces appear in Weave UI
   - Test trace hierarchy (parent-child relationships)
   - Verify no memory leaks from long-running traces

3. **Performance Tests**
   - Measure overhead of tracing
   - Test with high-volume scenarios (batch processing, API load)

## Success Criteria

1. All entry points have consistent tracing
2. Traces are visible in Weave UI with proper hierarchy
3. No performance degradation
4. Error scenarios are properly traced
5. Documentation updated with tracing guidelines

## Priority Order

1. **High Priority**
   - Batch processing modes (impacts research workflows)
   - API endpoints (production services)
   
2. **Medium Priority**
   - Slack bot handlers
   - Console mode
   
3. **Low Priority**
   - Streamlit interface
   - Development/debug endpoints

## Notes

- Consider adding trace sampling for high-volume endpoints
- WebSocket connections need special consideration for long-lived traces
- Batch jobs should have both job-level and item-level traces
- Consider adding trace context to error responses for debugging