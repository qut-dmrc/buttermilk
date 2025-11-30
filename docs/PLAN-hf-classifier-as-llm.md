# Plan: Treat HuggingFaceClassifier Like an LLM

## Context

The `gpt-oss-safeguard-20b` HuggingFace model is actually routed through Grok, which supports structured outputs. Currently, `HuggingFaceClassifier` bypasses the wrapper's schema support by passing `schema=None` and manually parsing JSON with `json.loads()`. This is fragile and inconsistent with how `LLMCore` handles structured outputs.

## Current State

### HuggingFaceClassifier ([classifier.py:287-518](buttermilk/agents/classifier.py#L287-L518))

1. **Gets wrapper**: `bm.llms.get_autogen_chat_client(self.model)` (line 328)
2. **Calls LLM with NO schema**: `await self._llm_wrapper.create(messages=messages, schema=None)` (line 380)
3. **Manual JSON parsing**: `json.loads(result.content)` (line 386)
4. **Custom mapping**: `_map_to_schema()` with hardcoded field mappings (lines 399-441)

### LLMCore ([llm_core.py:562-631](buttermilk/_core/llm_core.py#L562-L631))

1. **Gets wrapper**: `bm.llms.get_autogen_chat_client(self.model)` (line 591)
2. **Passes schema to wrapper**: `schema=self.output_model` (line 606)
3. **Wrapper handles parsing**: Returns `ModelOutput.parsed_object` already validated
4. **No manual mapping needed**: Gets Pydantic instance directly

### Wrapper Schema Handling ([llms.py:540-572](buttermilk/_core/llms.py#L540-L572))

The wrapper already supports two strategies:
1. **Native structured output**: If `model_info.structured_output=True`, passes `json_output=schema`
2. **Function calling fallback**: Creates fake `PydanticModelTool` to force schema compliance

## Proposed Changes

### 1. Update Model Configuration

The `gpt-oss-safeguard-20b` model info needs `structured_output: true` or at minimum `function_calling: true` to enable the wrapper's schema handling.

**Location**: Model connection secrets (GCP Secret Manager) or local test config

```yaml
gpt-oss-safeguard-20b:
  model_info:
    structured_output: true  # OR function_calling: true
```

### 2. Modify HuggingFaceClassifier._classify()

**Current** ([classifier.py:373-397](buttermilk/agents/classifier.py#L373-L397)):
```python
async def _classify(self, messages: list) -> dict[str, Any]:
    result = await self._llm_wrapper.create(messages=messages, schema=None)
    if isinstance(result.content, str):
        response = json.loads(result.content)  # Fragile!
    ...
```

**Proposed**:
```python
async def _classify(self, messages: list) -> BaseModel:
    """Call HuggingFace model via wrapper with structured output."""
    result = await self._llm_wrapper.create(
        messages=messages,
        schema=self.output_model  # Let wrapper handle parsing
    )

    # Wrapper returns ModelOutput with parsed_object
    if isinstance(result, ModelOutput) and result.parsed_object:
        return result.parsed_object  # Already validated Pydantic

    # Fallback: if wrapper couldn't parse, try manual
    if hasattr(result, "content") and result.content:
        return self._parse_raw_response(result.content)

    raise ProcessingError(f"Empty response from model: {result}")
```

### 3. Simplify _classify_record()

**Current** ([classifier.py:443-518](buttermilk/agents/classifier.py#L443-L518)):
```python
# Step 4: Map to schema
structured_output = self._map_to_schema(api_response, self.output_model)
```

**Proposed**:
```python
# Step 4: api_response is already a validated Pydantic model
# No mapping needed if wrapper parsed it
if isinstance(api_response, self.output_model):
    structured_output = api_response
else:
    # Fallback for legacy/incompatible responses
    structured_output = self._map_to_schema(api_response, self.output_model)
```

### 4. Keep _map_to_schema() as Fallback

Retain the existing mapping logic for:
- Models that truly don't support structured output
- Backward compatibility with HuggingFace-specific response formats (LABEL_0/LABEL_1)

### 5. Add Logging for Debugging

Add debug logging to show which path was taken:
- "Using native structured output"
- "Using function calling fallback"
- "Using manual JSON parsing fallback"

## Files to Modify

1. **[buttermilk/agents/classifier.py](buttermilk/agents/classifier.py)** - Main changes
   - `_classify()`: Pass `schema=self.output_model`
   - `_classify_record()`: Handle already-parsed result

2. **Model config** (if needed) - Enable structured_output or function_calling

## Testing Strategy

1. **Unit test**: Mock wrapper to return `ModelOutput` with `parsed_object`
2. **E2E test**: Run existing `test_classifier_e2e.py` to verify real API behavior
3. **Regression**: Ensure manual fallback still works for non-compliant models

## Benefits

1. **Reliability**: Wrapper validates schema; no fragile `json.loads()`
2. **Consistency**: Same pattern as LLMCore
3. **Maintainability**: Remove duplicate parsing logic
4. **Observability**: Wrapper's tracing captures structured output attempts

## Risks

1. **Model config**: Need to verify Grok actually supports structured output
2. **Backward compat**: Old responses might have different field names - fallback handles this
3. **Performance**: Function calling fallback adds overhead (but improves reliability)

## Next Steps

1. Verify model capabilities (does Grok support structured output natively?)
2. Update model config to enable structured_output
3. Implement changes to `_classify()` and `_classify_record()`
4. Run E2E tests to validate
