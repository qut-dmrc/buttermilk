# Actual Model Name Logging

## Summary

Changed the model logging system to capture and log the **actual model name returned by the API**, not just our internal shorthand name. This is essential when using model aliases like `"gemini-flash-latest"` or `"gemini-pro-latest"`, as we need to know which specific model version actually processed the request.

## Problem

Previously, when we configured models with aliases in `models.json`:

- We would log only our shorthand name (e.g., `"gemini25flash"`)
- We wouldn't know the actual model version that processed the request (e.g., `"gemini-2.0-flash-exp"`)
- This made it impossible to track which model versions were actually being used

## Solution

### Changes Made

#### 1. `buttermilk/_core/llms.py` - LiteLLM Response Conversion

**Lines 1007-1024**: Modified `litellm_to_autogen_result()` to extract and store the actual model name from LiteLLM responses:

```python
# Extract actual model name from response
# Prefer actual model from API (e.g., "gemini-2.0-flash-exp")
# Fall back to our shorthand if API doesn't provide it (e.g., "gemini25flash")
model_name = getattr(response, "model", model)

# Always return ModelOutput to preserve pricing metadata
result = ModelOutput(
    content=content,
    finish_reason=finish_reason,
    usage=request_usage,
    cached=cached,
    parsed_object=None,  # Will be parsed by caller if needed
)

# Store model name directly (actual from API or fallback to config name)
result.metadata["model"] = model_name
```

#### 2. `buttermilk/_core/llms.py` - AutoGenWrapper Model Extraction

**Lines 457-459**: Added extraction of model name from Autogen client responses:

```python
# Extract actual model name from response if available (some providers return this)
# Prefer actual model from API, fallback to our litellm_model_name
actual_model_name = getattr(create_result, "model", self.litellm_model_name)
```

**Lines 481-484, 541-544, 555-558, 570-573**: Updated all ModelOutput creation points to store model name directly:

```python
metadata = {
    "pricing": pricing_metadata,
    "model": actual_model_name,
}
```

#### 3. `buttermilk/_core/llm_core.py` - Metadata Collection

**Lines 437-449**: Simplified metadata collection to use model name from LLM wrapper:

```python
# Collect metadata (preserve existing template metadata)
# Model name comes from LLM wrapper (actual from API or config as fallback)
model_name = self.model  # Default to config name
if isinstance(llm_result, ModelOutput) and hasattr(llm_result, "metadata"):
    # Use model from wrapper (already contains actual API model or fallback)
    model_name = llm_result.metadata.get("model", self.model)

result.metadata = {
    **result.metadata,  # Keep template metadata added earlier
    "model": model_name,  # Actual model from API or config name as fallback
    "finish_reason": llm_result.finish_reason,
    "usage": llm_result.usage,
}
```

#### 4. `buttermilk/_core/llms.py` - LiteLLMWrapper Bug Fix

**Lines 1229-1231**: Fixed critical bug where metadata was being overwritten:

```python
# Add pricing metadata (result is always ModelOutput now)
# Preserve model that was set in litellm_to_autogen_result()
result.metadata["pricing"] = pricing_metadata
```

**Previous code** (incorrect):

```python
result.metadata = {"pricing": pricing_metadata}  # ❌ This overwrites all metadata!
```

## Metadata Structure

After these changes, the `"model"` field in metadata will contain:

- **The actual model name from the API** if available (e.g., `"gemini-2.0-flash-exp"`)
- **Our config/shorthand name** as a fallback (e.g., `"gemini25flash"`)

```python
{
    "model": "gemini-2.0-flash-exp",  # Real model from API (or config name if unavailable)
    "finish_reason": "stop",
    "usage": {...},
    "pricing": {...},
    "template": {...},
}
```

## Benefits

1. **Transparency**: Know exactly which model version processed each request
2. **Debugging**: Easier to track model behavior changes across versions
3. **Cost Tracking**: More accurate attribution when models have different pricing
4. **Auditing**: Full observability for compliance and analysis

## Provider Support

- ✅ **LiteLLM**: Fully supported via `response.model` field
- ⚠️ **Autogen Native Clients**: Support varies by provider (we attempt to extract but may fall back to config name)

## Next Steps

You can now safely update `models.json` to use model aliases:

```json
{
  "gemini25flash": {
    "client_type": "gemini",
    "model_info": {
      "vision": true,
      "family": "google-gemini",
      "model": "gemini-flash-latest" // ← Can use aliases now
    }
  }
}
```

The traces and logs will show both the alias and the actual resolved model name.

## Testing

A test script has been created at [test_actual_model_logging.py](test_actual_model_logging.py) to verify the functionality.

## Related Files

- [buttermilk/_core/llm_core.py](buttermilk/_core/llm_core.py#L437-L450)
- [buttermilk/_core/llms.py](buttermilk/_core/llms.py#L991-L1007)
