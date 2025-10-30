# OpenTelemetry Tracing Issues Investigation & Implementation Plan

**Date**: 2025-10-30 **Status**: Investigation Complete, Implementation Plan Ready **Priority**: High - Affects observability and trace analysis

## Executive Summary

Investigation of 8 critical OpenTelemetry (OTEL) tracing issues in Buttermilk's orchestrator flows. These issues affect trace nesting, session management, attribute capture, and downstream analysis in Google Cloud Trace.

**Key Findings**:

1. Trace lifecycle management issue causing improper nesting
1. Session ID mismatch between OTEL baggage and BM object
1. Service name configuration not propagated to traces
1. Agent metadata serialization producing verbose class names
1. Critical agent parameters missing from traces
1. Record injection lacks comprehensive tracing
1. Hash values not captured in tracing system
1. LLM call format incompatible with Google Cloud Trace parser

______________________________________________________________________

## Issue 1: Trace Nesting Problem

### Problem Statement

`buttermilk.flow.run` span is not ending properly, causing subsequent runs to nest under it instead of being separate traces. Either `buttermilk.flow.run` or `orchestrator.trans` should represent the start of a new session/trace.

### Root Cause Analysis

**Location**: `/home/nic/src/buttermilk/buttermilk/runner/flowrunner.py:1283-1294`

```python
# Wrap the flow execution in a tracing span using session-aware helper
with span_with_session(
    getattr(run_request, "session_id", None),
    name="buttermilk.flow.run",
    attributes={
        "buttermilk.flow.name": getattr(run_request, "flow", None),
        "buttermilk.job.id": getattr(run_request, "job_id", None),
        "buttermilk.source": ", ".join(run_request.source) if getattr(run_request, "source", None) else "direct",
        "buttermilk.mode": self.mode,
    },
    kind="internal",
) as span:
```

**Analysis**:

- `span_with_session()` is a context manager from `buttermilk/utils/otel.py:170-206`
- The context manager properly enters and exits, so the span SHOULD be ending
- However, there's also a **session root span** started at line 553 in `flowrunner.py`:
  ```python
  session._otel_session_root = start_session_root_span(
      session_id, attributes={"buttermilk.session.status": SessionStatus.INITIALIZING.value}
  )
  ```
- This session root span is created per session and only ends when the session is cleaned up
- **Issue**: Batch runs reuse the same session, so the session root span stays open across multiple flow runs
- This causes flow runs to nest under the session root span rather than being independent traces

**Evidence**:

1. `start_session_root_span()` in `otel.py:226-236` creates a long-lived span
1. It's only ended in `FlowRunContext.cleanup()` at `flowrunner.py:258-263`
1. Batch processing doesn't cleanup sessions between runs
1. Result: `buttermilk.session` span stays active, all flow runs nest under it

### Proposed Solution

**Option A: Remove Session Root Span (Recommended)**

- Remove the session root span entirely
- Use `buttermilk.flow.run` as the true root for each flow execution
- Session context propagated via baggage, not span hierarchy

**Option B: Proper Session Root Span Lifecycle**

- Create session root span ONLY for interactive/long-lived sessions (WebSocket)
- Skip session root span for batch jobs
- Each batch job gets independent `buttermilk.flow.run` root span

**Implementation Approach**: Option A

- **Rationale**: Session root span provides no additional value beyond baggage
- Baggage already propagates `buttermilk.session.id` to all child spans
- Removing it simplifies trace structure without losing session correlation

______________________________________________________________________

## Issue 2: Session ID Mismatch

### Problem Statement

There's a `session_id` in traces that differs from `buttermilk.session.id` in the bm object. Since the bm object is session-scoped, these IDs should match.

### Root Cause Analysis

**Multiple Session ID Sources**:

1. **OTEL Baggage**: `buttermilk.session.id` set via `attach_session_baggage()` in `otel.py:146-158`
1. **BM Session Info**: `bm.session_info.session_id` from `SessionInfo` in `bm_init.py:111`
1. **ExecutionTrace**: `session_id` field in `contract.py:533-536` using `_get_session_id()`

**Investigation**:

```python
# In otel.py:146-158
def attach_session_baggage(session_id: str | None, extra: dict | None = None) -> object | None:
    if not session_id:
        return None
    baggage = otel_baggage.set_baggage("buttermilk.session.id", session_id)
```

```python
# In contract.py:533-536
session_id: str = Field(
    default_factory=_get_session_id,
    description="Unique identifier for the client session or overall flow execution.",
)
```

```python
# In contract.py:77-86
def _get_session_id() -> str:
    """Get session_id from BM if available, otherwise generate new UUID."""
    try:
        from buttermilk._core.dmrc import get_bm
        bm = get_bm()
        return bm.session_info.session_id
    except Exception:
        # Fallback if BM not initialized
        return str(shortuuid.uuid())
```

**Problem Identified**:

- The code SHOULD be consistent because `_get_session_id()` reads from `bm.session_info.session_id`
- However, **timing issues** can occur:
  1. Session created in FlowRunner with session_id X
  1. BM object may have different session_id if it's global singleton vs session-scoped
  1. `attach_session_baggage()` called with FlowRunContext session_id
  1. `ExecutionTrace` created with BM's session_id (potentially different)

**Evidence**:

- In `flowrunner.py:1308`: Session created/retrieved with `run_request.session_id`
- In `flowrunner.py:521-522`: Baggage attached with `session_id` parameter
- But if using global BM singleton, `bm.session_info.session_id` might differ

### Proposed Solution

**Enforce Session-Scoped BM Usage**:

1. Always use session-scoped BM (already implemented via `FlowRunner.set_session_bm()`)
1. Ensure `run_request.session_id` always matches `bm.session_info.session_id`
1. Add validation in `FlowRunner.run_flow()` to verify match before execution
1. Update `_get_session_id()` to accept explicit session_id parameter to avoid BM lookup

**Implementation**:

```python
# In FlowRunner.run_flow()
def run_flow(self, run_request: RunRequest, ...):
    bm = self.get_effective_bm()

    # VALIDATION: Ensure session IDs match
    if bm.session_info.session_id != run_request.session_id:
        raise ValueError(
            f"Session ID mismatch: BM has {bm.session_info.session_id}, "
            f"request has {run_request.session_id}"
        )

    # Continue with execution...
```

______________________________________________________________________

## Issue 3: Service Name Not Used

### Problem Statement

The trace has a `service.name` field that should be set to `buttermilk's session_info.project_name` but isn't being used correctly.

### Root Cause Analysis

**Current Implementation**:

In `config_bootstrap.py:236-237`:

```python
"OTEL_SERVICE_NAME": otel_config.get("service_name", "buttermilk"),
"OTEL_RESOURCE_ATTRIBUTES": f"service.name={otel_config.get('service_name', 'buttermilk')}",
```

**Problems**:

1. Service name is hardcoded to `"buttermilk"` if not in config
1. It does NOT use `bm.session_info.project_name` as the user expected
1. Configuration is set globally in environment variables, not per-session

**Where Service Name Should Come From**:

- `bm.session_info.project_name` from `SessionInfo` (e.g., "dbr", "prosocial")
- This allows distinguishing traces by project in observability tools

**Why It's Not Working**:

- OTEL SDK reads `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` at initialization time
- These are set once during bootstrap, before any session is created
- Resource attributes are **global** to the tracer provider, not per-span

### Proposed Solution

**Use Span Attributes Instead of Resource Attributes**:

1. Keep `service.name` resource attribute as "buttermilk" (identifies the application)
1. Add `project.name` as a **span attribute** on each span
1. This allows per-session project tracking while maintaining global service identity

**Implementation**:

```python
# In otel.py - Update span_with_session() to include project name
def span_with_session(
    session_id: str | None,
    name: str,
    attributes: dict | None = None,
    kind: str | _SpanKind | None = None,
):
    # Get project name from BM
    from buttermilk._core.dmrc import get_bm
    try:
        bm = get_bm()
        project_name = bm.session_info.project_name
    except Exception:
        project_name = "unknown"

    # Merge attributes with session id AND project name
    base_attrs = {
        "buttermilk.session.id": session_id,
        "buttermilk.project.name": project_name,  # ADD THIS
    } if session_id else {}
    all_attrs = _clean_attrs({**base_attrs, **(attributes or {})})
    # ... rest of implementation
```

**Alternative**: Add to BaggageToAttributesSpanProcessor keys:

```python
provider.add_span_processor(
    BaggageToAttributesSpanProcessor(keys=[
        "buttermilk.session.id",
        "buttermilk.project",  # ADD THIS
        "buttermilk.execution_context.id",
    ])
)
```

______________________________________________________________________

## Issue 4: Agent Type Formatting

### Problem Statement

`agent.type` is written as `<class 'buttermilk.agents.judge.Judge'>` - need to determine if this is best practice or should be simplified.

### Root Cause Analysis

**Location**: Agent tracing likely in `agent.py` or orchestrator's agent execution

**Current Behavior**:

- Using `type(agent)` or `agent.__class__` directly produces: `<class 'buttermilk.agents.judge.Judge'>`
- This is Python's default `repr()` for class objects

**Investigation of Agent Info Capture**:

```python
# In contract.py:543-546
agent_info: dict[str, Any] = Field(
    ...,  # Mandatory field
    description="Component configuration and metadata (component_name, execution_type, config, etc.).",
)
```

**Search for agent.type**:

- Found in `agent.py` and `groupchat.py`
- Likely set during ExecutionTrace creation

**Best Practices**:

- OTEL Semantic Conventions recommend using simple strings for component types
- Example: `"judge"`, `"fetch"`, `"synth"` instead of full class path
- Full class path useful for debugging but verbose in traces

### Proposed Solution

**Use Simple Agent Type Names**:

```python
# Instead of: str(type(agent))
# Use: agent.__class__.__name__  -> "Judge"
# Or even better: agent.role  -> "JUDGE" (from config)
```

**Implementation**:

```python
def get_agent_type_for_trace(agent) -> str:
    """Get simplified agent type for tracing.

    Returns lowercase agent class name (e.g., 'judge', 'fetch').
    """
    return agent.__class__.__name__.lower()
```

**Add to ExecutionTrace Creation**:

```python
agent_info = {
    "agent_type": get_agent_type_for_trace(agent),  # Simple name
    "agent_class": f"{agent.__class__.__module__}.{agent.__class__.__name__}",  # Full path if needed
    "agent_role": agent.role,  # Role from config (e.g., "JUDGE")
    # ... other fields
}
```

______________________________________________________________________

## Issue 5: Missing Agent Parameters

### Problem Statement

Critical agent parameters are missing from traces:

- template
- template_hash
- model

### Root Cause Analysis

**Agent Configuration Structure** (from `config.py` and `agent.py`):

```python
class AgentConfig:
    agent_id: str
    agent_name: str
    role: str
    description: str
    parameters: dict[str, Any]  # This contains template, model, etc.
    inputs: dict[str, str] | None
```

**Current Trace Capture**:

- ExecutionTrace has `agent_info` dict
- But it's not clear what gets populated in `agent_info`

**What Should Be Captured**:

1. **template**: The Jinja2 template name/path used by the agent
1. **template_hash**: Hash of the template for reproducibility tracking
1. **model**: LLM model used (e.g., "gemini-2.0-flash-thinking-exp")

**Where These Values Live**:

- `agent.parameters["template"]` - Template name
- `agent.parameters["model"]` - Model identifier
- Template hash computed in `hashing.py` or `templating.py`

**Search Results**:

- `template_hash` found in: `llm_core.py`, `hashing.py`, `templating.py`
- These are critical for experiment reproducibility

### Proposed Solution

**Enhance Agent Info Capture in ExecutionTrace**:

```python
def create_agent_trace_info(agent, template_hash: str | None = None) -> dict:
    """Create comprehensive agent info for ExecutionTrace.

    Captures all critical parameters for reproducibility and debugging.
    """
    agent_info = {
        # Identity
        "agent_type": agent.__class__.__name__.lower(),
        "agent_name": agent.agent_name,
        "agent_role": agent.role,

        # Critical parameters
        "template": agent.parameters.get("template"),
        "template_hash": template_hash or agent.parameters.get("template_hash"),
        "model": agent.parameters.get("model"),

        # Full config (for reference)
        "parameters": agent.parameters,

        # Additional metadata
        "description": agent.description,
    }

    return {k: v for k, v in agent_info.items() if v is not None}
```

**Integration Points**:

1. Update agent execution to compute template_hash if not already present
1. Pass template_hash to ExecutionTrace creation
1. Ensure model name is consistently captured from LLM config

______________________________________________________________________

## Issue 6: Record Injection Tracing

### Problem Statement

Records should get their own trace when injected into a groupchat. Currently there's a trace from Fetch, but it lacks:

- Important record attributes
- Hash values

### Root Cause Analysis

**Record Injection Flow**:

1. Fetch agent retrieves records from storage
1. Records are "injected" into groupchat context
1. Other agents process these records

**Current Tracing**:

- Fetch agent likely creates ExecutionTrace for its own execution
- But record injection as a distinct operation may not be traced

**What Should Be Traced**:

```python
{
    "operation": "record.inject",
    "record_id": record.record_id,
    "record_hash": record.record_hash,
    "record_type": type(record).__name__,
    "attributes": {
        # Record-specific attributes
        "source": record.source,
        "dataset": record.dataset,
        # Any other important record metadata
    },
    "hashes": {
        "record_hash": record.record_hash,
        "content_hash": hash(record.content),
    }
}
```

### Proposed Solution

**Create Record Injection Trace**:

```python
async def inject_record_with_trace(
    record: BaseRecord,
    session_id: str,
    parent_call_id: str | None = None
) -> None:
    """Inject record into groupchat with proper tracing."""

    from buttermilk.utils.otel import span_with_session

    # Create span for record injection
    with span_with_session(
        session_id,
        name="record.inject",
        attributes={
            "record.id": record.record_id,
            "record.type": type(record).__name__,
            "record.hash": getattr(record, "record_hash", None),
            "record.source": getattr(record, "source", None),
            "record.dataset": getattr(record, "dataset", None),
        },
        kind="internal"
    ) as span:
        # Perform actual injection
        await groupchat.inject_record(record)

        # Log additional hash values
        if span and hasattr(record, "content"):
            span.set_attribute("record.content_hash", hash_content(record.content))
```

**Integration**:

- Add this to orchestrator's record injection logic
- Ensure Fetch agent provides all necessary record metadata

______________________________________________________________________

## Issue 7: Hash Logging

### Problem Statement

Every hash in the system should be logged in traces.

### Root Cause Analysis

**Hash Types in Buttermilk**:

From `hashing.py` and search results:

1. **template_hash**: Hash of Jinja2 template content
1. **record_hash**: Hash of record content
1. **config_hash**: Hash of agent configuration
1. **output_hash**: Hash of agent output
1. **model_hash**: Hash of model configuration

**Current State**:

- Some hashes computed but not consistently traced
- Template hash in `templating.py` and `llm_core.py`
- Record hash likely in BaseRecord
- No systematic hash capture in ExecutionTrace

**Why Hashes Matter**:

- **Reproducibility**: Detect config/template changes
- **Deduplication**: Identify duplicate executions
- **Caching**: Enable result caching based on input hash
- **Debugging**: Track exactly what configuration was used

### Proposed Solution

**Systematic Hash Capture Strategy**:

1. **Define Hash Namespace**:

```python
# Standard hash attributes for OTEL spans
HASH_ATTRIBUTES = {
    "hash.template": str,      # Template content hash
    "hash.config": str,        # Agent config hash
    "hash.input": str,         # Input data hash
    "hash.output": str,        # Output data hash
    "hash.record": str,        # Record content hash
    "hash.model_config": str,  # LLM config hash
}
```

2. **Enhance Hashing Module**:

```python
# In hashing.py
class HashCollector:
    """Collects all hashes for a trace."""

    def __init__(self):
        self.hashes: dict[str, str] = {}

    def add_template_hash(self, template_name: str, template_content: str):
        self.hashes["hash.template"] = hash_template(template_content)
        self.hashes["hash.template.name"] = template_name

    def add_config_hash(self, config: dict):
        self.hashes["hash.config"] = hash_dict(config)

    def add_record_hash(self, record: BaseRecord):
        self.hashes["hash.record"] = record.record_hash
        self.hashes["hash.record.id"] = record.record_id

    def to_span_attributes(self) -> dict:
        """Convert to OTEL span attributes."""
        return {k: v for k, v in self.hashes.items() if v is not None}
```

3. **Integration in Agent Execution**:

```python
# When creating ExecutionTrace
hash_collector = HashCollector()
hash_collector.add_template_hash(template_name, template_content)
hash_collector.add_config_hash(agent.parameters)

trace = ExecutionTrace(
    agent_info={
        **agent_base_info,
        **hash_collector.to_span_attributes(),
    },
    # ... other fields
)
```

4. **Add to OTEL Spans**:

```python
# In span creation
with span_with_session(
    session_id,
    name="agent.execute",
    attributes={
        **standard_attributes,
        **hash_collector.to_span_attributes(),
    }
) as span:
    # Execute agent
```

______________________________________________________________________

## Issue 8: LLM Call Format Issue

### Problem Statement

LLM calls appear formatted according to OTEL GenAI standard, but Google traces cannot parse them.

### Root Cause Analysis

**OTEL GenAI Instrumentation**:

From `otel.py:98-108`:

```python
# Instrument libraries (only if not already instrumented)
if not OpenAIInstrumentor().is_instrumented_by_opentelemetry:
    OpenAIInstrumentor().instrument(tracer_provider=provider)
if not GoogleGenerativeAiInstrumentor().is_instrumented_by_opentelemetry:
    GoogleGenerativeAiInstrumentor().instrument(tracer_provider=provider)
```

**OTEL GenAI Semantic Conventions**:

- Standard attributes: `gen_ai.request.model`, `gen_ai.response.model`, etc.
- Events for streaming: `gen_ai.content.prompt`, `gen_ai.content.completion`
- Google Cloud Trace has its own expected format

**The Problem**:

- OTEL GenAI spec creates spans/events in a generic format
- Google Cloud Trace expects specific format (possibly Vertex AI or Cloud Logging format)
- Mismatch causes parsing failures in Google Cloud Console

**Investigation Needed**:

- What specific format does Google Cloud Trace expect?
- Are there examples of working LLM traces in Google Cloud Trace?
- Is there Google-specific instrumentation we should use instead?

### Proposed Solution

**Option A: Use Google-Specific Instrumentation**

```python
# Instead of generic OTEL GenAI instrumentors
# Use Google Cloud-specific instrumentation
from google.cloud.aiplatform.telemetry import instrumentation

# Or custom span formatting for Vertex AI
```

**Option B: Custom Span Processor for Google Format**

```python
class GoogleTraceFormatProcessor(SpanProcessor):
    """Reformats GenAI spans to Google Cloud Trace format."""

    def on_end(self, span):
        # Transform span attributes to Google-expected format
        if span.attributes.get("gen_ai.request.model"):
            # Reformat to Google's expected structure
            span.set_attribute("model.name", span.attributes["gen_ai.request.model"])
            # ... other transformations
```

**Option C: Use Vertex AI SDK's Built-in Tracing**

```python
# Vertex AI SDK may have built-in tracing that works with Google Cloud Trace
from vertexai import generative_models
from vertexai.telemetry import telemetry

# Use Vertex SDK's telemetry instead of OTEL GenAI
```

**Recommendation**:

- **Investigate** Google Cloud Trace documentation for LLM tracing
- **Test** current traces in Google Cloud Console to see exact error
- **Implement** Option C if Vertex AI SDK provides compatible tracing
- **Fall back** to Option B if we need custom formatting

______________________________________________________________________

## Implementation Plan

### Phase 1: Session & Trace Lifecycle (Issues 1 & 2)

**Priority**: P0 (Highest) **Estimated Time**: 3-4 days

#### Tasks:

1. **Remove Session Root Span** (Issue 1)

   - [ ] Remove `start_session_root_span()` call from `FlowRunContext.get_or_create_session()`
   - [ ] Remove `end_session_root_span()` call from cleanup
   - [ ] Update tests that expect session root span
   - [ ] Verify batch runs create independent traces

1. **Fix Session ID Consistency** (Issue 2)

   - [ ] Add session ID validation in `FlowRunner.run_flow()`
   - [ ] Ensure session-scoped BM is used consistently
   - [ ] Update `_get_session_id()` to accept explicit parameter
   - [ ] Add unit tests for session ID matching

#### Success Criteria:

- [ ] Batch runs create separate top-level traces (not nested)
- [ ] `buttermilk.session.id` matches `bm.session_info.session_id` in all traces
- [ ] Tests pass: `test_batch_trace_independence`, `test_session_id_consistency`

#### Testing Strategy:

```python
# Test batch run trace independence
async def test_batch_runs_create_independent_traces():
    """Verify each batch run creates a separate root trace."""
    batch = await flow_runner.create_batch("test_flow", max_records=3)

    # Run batch
    await flow_runner.run_batch_job(...)

    # Verify traces
    traces = get_traces_from_otel()
    root_traces = [t for t in traces if t.parent_span_id is None]

    assert len(root_traces) == 3  # Each run is independent
    assert all(t.name == "buttermilk.flow.run" for t in root_traces)

# Test session ID consistency
async def test_session_id_consistency():
    """Verify session IDs match across BM and OTEL baggage."""
    session_id = "test-session-123"
    request = RunRequest(session_id=session_id, flow="test")

    await flow_runner.run_flow(request, wait_for_completion=True)

    traces = get_traces_for_session(session_id)
    assert all(t.attributes["buttermilk.session.id"] == session_id for t in traces)
```

______________________________________________________________________

### Phase 2: Attribute Capture (Issues 3, 4, 5, 6, 7)

**Priority**: P1 **Estimated Time**: 5-6 days

#### Tasks:

1. **Add Project Name to Spans** (Issue 3)

   - [ ] Update `span_with_session()` to include `buttermilk.project.name`
   - [ ] Add to `BaggageToAttributesSpanProcessor` keys
   - [ ] Verify project name appears in all child spans

1. **Fix Agent Type Formatting** (Issue 4)

   - [ ] Create `get_agent_type_for_trace()` utility
   - [ ] Update ExecutionTrace creation to use simple agent type
   - [ ] Keep full class path as separate attribute
   - [ ] Update tests expecting old format

1. **Capture Agent Parameters** (Issue 5)

   - [ ] Create `create_agent_trace_info()` function
   - [ ] Integrate template_hash computation
   - [ ] Capture model, template, and other critical parameters
   - [ ] Add to ExecutionTrace agent_info

1. **Add Record Injection Tracing** (Issue 6)

   - [ ] Create `inject_record_with_trace()` function
   - [ ] Integrate into orchestrator record injection flow
   - [ ] Capture record hash, attributes, and metadata
   - [ ] Add tests for record injection traces

1. **Systematic Hash Logging** (Issue 7)

   - [ ] Create `HashCollector` class in `hashing.py`
   - [ ] Integrate hash collection in agent execution
   - [ ] Add hash attributes to OTEL spans
   - [ ] Document hash naming conventions

#### Success Criteria:

- [ ] All traces include `buttermilk.project.name` attribute
- [ ] Agent type is simple name (e.g., "judge") not full class
- [ ] All critical parameters (template, model, hashes) captured
- [ ] Record injections create separate trace spans
- [ ] All hashes logged with consistent naming

#### Testing Strategy:

```python
# Test agent parameter capture
async def test_agent_parameters_in_trace():
    """Verify critical agent parameters are captured."""
    request = RunRequest(flow="test_flow")
    await flow_runner.run_flow(request, wait_for_completion=True)

    traces = get_agent_traces()
    judge_trace = next(t for t in traces if t.agent_info["agent_type"] == "judge")

    assert "template" in judge_trace.agent_info
    assert "template_hash" in judge_trace.agent_info
    assert "model" in judge_trace.agent_info
    assert judge_trace.agent_info["model"] is not None

# Test hash capture
async def test_hash_logging():
    """Verify all hashes are logged."""
    traces = await run_flow_and_get_traces()

    for trace in traces:
        if trace.agent_info["agent_type"] == "judge":
            assert "hash.template" in trace.attributes
            assert "hash.config" in trace.attributes
            assert "hash.record" in trace.attributes
```

______________________________________________________________________

### Phase 3: LLM Call Format Investigation (Issue 8)

**Priority**: P2 **Estimated Time**: 3-5 days (includes research)

#### Research Tasks:

1. **Investigate Google Cloud Trace Format**

   - [ ] Review Google Cloud Trace documentation for LLM traces
   - [ ] Examine working examples (if any exist)
   - [ ] Identify expected span/event structure
   - [ ] Document format requirements

1. **Test Current Implementation**

   - [ ] Run flow with LLM calls
   - [ ] Export traces to Google Cloud Trace
   - [ ] Capture exact parsing errors
   - [ ] Document discrepancies

1. **Prototype Solutions**

   - [ ] Test Vertex AI SDK built-in tracing
   - [ ] Implement custom span processor if needed
   - [ ] Compare approaches
   - [ ] Choose best solution

#### Implementation Tasks:

1. **Implement Chosen Solution**

   - [ ] Code changes based on research
   - [ ] Add configuration for format selection
   - [ ] Update documentation
   - [ ] Integration tests

1. **Validation**

   - [ ] Verify traces parse correctly in Google Cloud Trace
   - [ ] Check trace visualization works
   - [ ] Ensure no data loss
   - [ ] Performance testing

#### Success Criteria:

- [ ] LLM call traces visible and parseable in Google Cloud Trace
- [ ] No parsing errors in Cloud Trace console
- [ ] All LLM metadata captured correctly
- [ ] Performance impact < 5%

#### Testing Strategy:

```python
# Test Google Cloud Trace compatibility
async def test_llm_traces_in_google_cloud():
    """Verify LLM traces are compatible with Google Cloud Trace."""
    # Run flow with LLM call
    request = RunRequest(flow="llm_flow")
    await flow_runner.run_flow(request, wait_for_completion=True)

    # Export to Google Cloud Trace
    export_traces_to_gcp()

    # Verify via Cloud Trace API
    from google.cloud import trace_v2
    client = trace_v2.TraceServiceClient()

    traces = client.list_traces(project_id=PROJECT_ID)
    llm_spans = [s for s in traces if "gen_ai" in s.name or "llm" in s.name]

    assert len(llm_spans) > 0, "No LLM spans found"
    assert all(s.status.code == 0 for s in llm_spans), "Parse errors found"
```

______________________________________________________________________

## Testing & Validation

### Unit Tests

```python
# tests/unit/test_otel_tracing_fixes.py

class TestTraceLifecycle:
    """Test trace nesting and session management."""

    async def test_batch_runs_independent_traces(self):
        """Each batch run creates independent root trace."""
        # Test implementation from Phase 1

    async def test_session_id_consistency(self):
        """Session IDs match across BM and OTEL."""
        # Test implementation from Phase 1

class TestAttributeCapture:
    """Test all attributes are captured correctly."""

    async def test_project_name_in_spans(self):
        """Project name appears in all spans."""
        # Test from Phase 2

    async def test_agent_parameters_captured(self):
        """Critical agent parameters in traces."""
        # Test from Phase 2

    async def test_hash_logging(self):
        """All hashes logged consistently."""
        # Test from Phase 2

class TestLLMTracing:
    """Test LLM call tracing compatibility."""

    async def test_google_cloud_trace_compatibility(self):
        """LLM traces parse in Google Cloud Trace."""
        # Test from Phase 3
```

### Integration Tests

```python
# tests/endtoend/test_complete_trace_lifecycle.py

async def test_end_to_end_trace_flow():
    """Test complete flow execution with all tracing enhancements."""

    # Setup
    flow_runner = await setup_flow_runner()
    session_bm = await create_session_bm(
        project_name="test_project",
        job="test_job"
    )
    flow_runner.set_session_bm(session_bm)

    # Execute flow
    request = RunRequest(
        flow="trans",
        session_id=session_bm.session_info.session_id,
        inputs={"record_id": "test-123"}
    )
    await flow_runner.run_flow(request, wait_for_completion=True)

    # Validate traces
    traces = get_all_traces()

    # Check trace structure
    root_traces = [t for t in traces if not t.parent_span_id]
    assert len(root_traces) == 1
    assert root_traces[0].name == "buttermilk.flow.run"

    # Check session IDs
    assert all(
        t.attributes["buttermilk.session.id"] == session_bm.session_info.session_id
        for t in traces
    )

    # Check project name
    assert all(
        t.attributes["buttermilk.project.name"] == "test_project"
        for t in traces
    )

    # Check agent traces
    agent_traces = [t for t in traces if "agent" in t.name]
    for trace in agent_traces:
        assert "agent_type" in trace.agent_info
        assert not trace.agent_info["agent_type"].startswith("<class")
        assert "template" in trace.agent_info
        assert "model" in trace.agent_info
        assert "hash.template" in trace.attributes

    # Check record injection
    record_traces = [t for t in traces if t.name == "record.inject"]
    assert len(record_traces) > 0
    for trace in record_traces:
        assert "record.id" in trace.attributes
        assert "record.hash" in trace.attributes
```

______________________________________________________________________

## Rollout Plan

### Stage 1: Development & Testing (Week 1-2)

- Implement Phase 1 (Session & Trace Lifecycle)
- Unit tests for trace independence and session ID consistency
- Code review and iteration

### Stage 2: Attribute Capture (Week 2-3)

- Implement Phase 2 (Attribute Capture)
- Integration tests for complete attribute coverage
- Performance testing to ensure < 5% overhead

### Stage 3: LLM Format Investigation (Week 3-4)

- Research Google Cloud Trace format requirements
- Prototype and test solutions
- Implement chosen approach

### Stage 4: End-to-End Testing (Week 4)

- Run complete flow executions with all fixes
- Verify traces in Google Cloud Trace console
- Load testing with batch runs
- Documentation updates

### Stage 5: Production Rollout (Week 5)

- Deploy to staging environment
- Monitor traces for 2-3 days
- Fix any issues discovered
- Deploy to production
- Monitor and iterate

______________________________________________________________________

## Risks & Mitigation

### Risk 1: Breaking Existing Trace Analysis

**Impact**: High **Probability**: Medium **Mitigation**:

- Keep old trace format side-by-side during transition
- Add feature flag for new tracing system
- Gradual rollout with canary deployments

### Risk 2: Performance Degradation

**Impact**: Medium **Probability**: Low **Mitigation**:

- Benchmark hash computation overhead
- Use async hash computation where possible
- Implement sampling for high-volume traces

### Risk 3: Google Cloud Trace Incompatibility

**Impact**: High **Probability**: Medium (for Issue 8) **Mitigation**:

- Early testing with Google Cloud Trace
- Fallback to basic span format if needed
- Consult Google Cloud support if issues persist

### Risk 4: Session ID Mismatch Edge Cases

**Impact**: Medium **Probability**: Low **Mitigation**:

- Comprehensive validation before execution
- Clear error messages when mismatch detected
- Logging for debugging session creation

______________________________________________________________________

## Success Metrics

### Objective Metrics

1. **Trace Independence**: 100% of batch runs create independent root traces
1. **Session ID Match**: 100% consistency between BM and OTEL session IDs
1. **Attribute Coverage**: 100% of traces include project_name, agent parameters, hashes
1. **Parse Success**: 100% of traces parseable in Google Cloud Trace
1. **Performance**: < 5% overhead from tracing enhancements

### Subjective Metrics

1. **Developer Experience**: Easier debugging with comprehensive trace data
1. **Trace Analysis**: Faster identification of issues in Google Cloud Trace
1. **Reproducibility**: Ability to recreate exact execution from trace data

______________________________________________________________________

## Future Enhancements

### Post-Implementation Improvements

1. **Trace Sampling**: Implement intelligent sampling for high-volume environments
1. **Trace Compression**: Compress large attribute values
1. **Trace Querying**: Build custom trace query tools
1. **Trace Visualization**: Custom visualization for Buttermilk-specific traces
1. **Trace Alerts**: Automated alerts based on trace patterns

______________________________________________________________________

## Appendix

### Key Files Modified

1. **`buttermilk/runner/flowrunner.py`**

   - Remove session root span
   - Add session ID validation
   - Update trace attribute capture

1. **`buttermilk/utils/otel.py`**

   - Update `span_with_session()` for project name
   - Remove `start_session_root_span()` (or deprecate)
   - Add baggage keys for project

1. **`buttermilk/_core/contract.py`**

   - Update `ExecutionTrace.agent_info` structure
   - Add hash fields
   - Document expected attributes

1. **`buttermilk/_core/hashing.py`**

   - Create `HashCollector` class
   - Standardize hash naming
   - Add hash utilities

1. **`buttermilk/_core/agent.py`**

   - Update trace creation
   - Capture agent parameters
   - Add hash computation

1. **`buttermilk/orchestrators/groupchat.py`**

   - Add record injection tracing
   - Update agent execution tracing

### References

- [OTEL Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [OTEL GenAI Spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Google Cloud Trace Docs](https://cloud.google.com/trace/docs)
- [Buttermilk Tracing Architecture](./DEBUGGING.md)

______________________________________________________________________

**Document Version**: 1.0 **Last Updated**: 2025-10-30 **Next Review**: After Phase 1 completion
