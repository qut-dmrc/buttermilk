# LiteLLM Migration Guide

This guide explains how to migrate your LLM configurations from Autogen to LiteLLM for unified, provider-agnostic LLM API calls.

## Overview

The LiteLLM integration provides a drop-in replacement for Autogen's ChatCompletionClient with support for 100+ LLM providers through a unified interface. Both wrappers (`AutoGenWrapper` and `LiteLLMWrapper`) provide identical APIs, ensuring zero breaking changes.

## Benefits

✅ **Provider flexibility**: Support for 100+ providers out of the box ✅ **Simplified configuration**: Uniform interface across all providers ✅ **Better cost tracking**: Already using LiteLLM for pricing ✅ **Gradual migration**: Test per-model without breaking existing flows ✅ **Zero breaking changes**: Existing API stays fully intact ✅ **Reduced custom code**: Less provider-specific credential handling

## Quick Start

### 1. Enable LiteLLM for a Model

To enable LiteLLM for a specific model, add `"use_litellm": true` to the model configuration in your `models.json` file:

```json
{
  "gemini25flash": {
    "client_type": "gemini_vertex",
    "model_info": {
      "vision": true,
      "function_calling": true,
      "json_output": true,
      "structured_output": false,
      "family": "gemini-2.5-flash"
    },
    "configs": {
      "model": "gemini-2.5-flash-preview-05-20",
      "temperature": 0.7,
      "region": "us-central1",
      "project_id": "your-project-id"
    },
    "use_litellm": true
  }
}
```

### 2. No Code Changes Required

The migration is completely transparent to your existing code. All existing code using `bm.llms.get_autogen_chat_client()` continues to work:

```python
# This works with both AutoGenWrapper and LiteLLMWrapper
client = bm.llms.get_autogen_chat_client("gemini25flash")

# Same interface for both wrappers
result = await client.create(
    messages=[UserMessage(content="Hello!", source="user")],
    schema=MySchema,  # Optional structured output
)
```

## Configuration Examples

### OpenAI (Direct API)

```json
{
  "gpt4": {
    "client_type": "openai",
    "api_key": "${OPENAI_API_KEY}",
    "model_info": {
      "vision": false,
      "function_calling": true,
      "json_output": true,
      "structured_output": true,
      "family": "gpt-4"
    },
    "configs": {
      "model": "gpt-4",
      "temperature": 0.7
    },
    "use_litellm": true
  }
}
```

### Azure OpenAI

```json
{
  "azure_gpt4": {
    "client_type": "azure",
    "api_key": "${AZURE_OPENAI_API_KEY}",
    "base_url": "https://your-resource.openai.azure.com/",
    "model_info": {
      "vision": false,
      "function_calling": true,
      "json_output": true,
      "structured_output": true,
      "family": "gpt-4"
    },
    "configs": {
      "model": "gpt-4",
      "api_version": "2024-02-15-preview"
    },
    "use_litellm": true
  }
}
```

### Anthropic (Direct API)

```json
{
  "claude": {
    "client_type": "anthropic",
    "api_key": "${ANTHROPIC_API_KEY}",
    "model_info": {
      "vision": false,
      "function_calling": true,
      "json_output": false,
      "structured_output": false,
      "family": "claude-3"
    },
    "configs": {
      "model": "claude-3-5-sonnet-20241022",
      "max_tokens": 4096
    },
    "use_litellm": true
  }
}
```

### Anthropic via Vertex AI

```json
{
  "claude_vertex": {
    "client_type": "anthropic_vertex",
    "model_info": {
      "vision": false,
      "function_calling": true,
      "json_output": false,
      "structured_output": false,
      "family": "claude-3"
    },
    "configs": {
      "model": "claude-3-5-sonnet-v2@20241022",
      "region": "us-central1",
      "project_id": "your-project-id"
    },
    "use_litellm": true
  }
}
```

### Google Gemini via Vertex AI

```json
{
  "gemini_pro": {
    "client_type": "gemini_vertex",
    "model_info": {
      "vision": true,
      "function_calling": true,
      "json_output": true,
      "structured_output": false,
      "family": "gemini-2.0-flash"
    },
    "configs": {
      "model": "gemini-2.0-flash-exp",
      "region": "us-central1",
      "project_id": "your-project-id"
    },
    "use_litellm": true
  }
}
```

## Feature Compatibility

Both `AutoGenWrapper` and `LiteLLMWrapper` support:

| Feature                      | AutoGenWrapper | LiteLLMWrapper | Notes                  |
| ---------------------------- | -------------- | -------------- | ---------------------- |
| Basic completions            | ✅             | ✅             | Full parity            |
| Structured output (Pydantic) | ✅             | ✅             | Full parity            |
| Tool/function calling        | ✅             | ✅             | Full parity            |
| Retry logic                  | ✅             | ✅             | Exponential backoff    |
| Pricing calculation          | ✅             | ✅             | Uses LiteLLM           |
| Weave tracing                | ✅             | ✅             | `@weave.op` decorators |
| OpenTelemetry spans          | ✅             | ✅             | Full observability     |
| ExecutionTrace               | ✅             | ✅             | Identical metadata     |

## Migration Strategy

### Phase 1: Test with One Model

Start by enabling LiteLLM for a single, low-risk model:

1. Choose a model that is not mission-critical (e.g., `gemini25flash`)
1. Add `"use_litellm": true` to its configuration
1. Run your existing flows and monitor for any issues
1. Verify that ExecutionTrace metadata looks correct
1. Check that pricing calculations are accurate

### Phase 2: Gradual Rollout

Once confident with one model:

1. Enable LiteLLM for 2-3 more models
1. Monitor production usage and observability
1. Gradually expand to more models
1. Document any provider-specific quirks

### Phase 3: Complete Migration (Optional)

Eventually migrate all models:

1. Enable LiteLLM for all remaining models
1. Verify all flows work correctly
1. Consider deprecating AutoGenWrapper (optional)

## Troubleshooting

### LiteLLM Not Available Error

```
ImportError: LiteLLM is not installed. Please install it with: pip install litellm
```

**Solution**: LiteLLM is already in project dependencies. This error indicates an environment issue. Verify your virtual environment is set up correctly:

```bash
uv sync
```

### Provider-Specific Authentication Issues

If you encounter authentication errors:

1. **OpenAI/Azure**: Verify `api_key` is set correctly
1. **Vertex AI**: Ensure GCP credentials are available via `bm.gcp_credentials`
1. **Anthropic**: Check that `ANTHROPIC_API_KEY` environment variable is set

### Model Name Resolution

LiteLLM uses specific model name formats. The `litellm_model` field in config allows overriding:

```json
{
  "my_model": {
    "client_type": "azure",
    "configs": {
      "model": "gpt-4"
    },
    "litellm_model": "azure/gpt-4", // Override model name for both API calls and pricing
    "use_litellm": true
  }
}
```

### Structured Output Not Working

Some models don't support native structured output. LiteLLMWrapper will:

1. Try `response_format` if `model_info.structured_output` is true
1. Fall back to JSON mode parsing if not supported
1. Return error in `result.error_message` if parsing fails

## Observability

Both wrappers maintain identical observability:

### Weave Tracing

```python
# Automatically traced with @weave.op
result = await client.create(messages=[...])

# Weave call ID available in ExecutionTrace
trace.call_id  # Unique ID
trace.parent_call_id  # Parent span ID
```

### OpenTelemetry

```python
# Spans are automatically created
# - "llm_core.call_llm" for LLM calls
# - "llm_core.process" for processing
# Token usage recorded in span attributes
```

### ExecutionTrace

```python
# Same metadata structure for both wrappers
trace = ExecutionTrace(
    call_id=result.trace_id,
    agent_info={
        "component_name": "LLMCore",
        "execution_type": "llm_processing",
        ...
    },
    metadata={
        "model": "gpt-4",
        "usage": {...},
        "pricing": {...},
        "duration_ms": 1234
    }
)
```

## Advanced Configuration

### Retry Configuration

Customize retry behavior per model:

```python
from buttermilk._core.llms import LiteLLMWrapper

# Custom retry settings (applied when creating wrapper)
wrapper = LiteLLMWrapper(
    model="gpt-4",
    model_info=model_info,
    litellm_model_name="gpt-4",
    max_retries=5,  # Default: 3
    min_wait_seconds=10.0,  # Default: 5.0
    max_wait_seconds=120.0,  # Default: 60.0
    jitter_seconds=10.0,  # Default: 5.0
)
```

### Custom Parameters

Pass additional parameters to LiteLLM:

```python
# Via configuration
{
    "my_model": {
        "configs": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
        },
        "use_litellm": true,
    }
}
```

## Performance Considerations

### Latency

LiteLLM adds minimal overhead (~1-2ms) for message format conversion. The actual API call latency is identical to Autogen.

### Caching

Both wrappers cache instantiated clients in `bm.llms.autogen_models`. Subsequent calls reuse the same wrapper instance.

### Concurrency

Both wrappers support concurrent requests. Use `asyncio.gather()` for parallel calls:

```python
import asyncio

tasks = [
    client.create(messages=[...]),
    client.create(messages=[...]),
    client.create(messages=[...]),
]

results = await asyncio.gather(*tasks)
```

## Backward Compatibility

The migration maintains 100% backward compatibility:

❌ **No changes required to**:

- Existing agent code
- LLMCore implementation
- ExecutionTrace structure
- Template rendering
- Tool calling logic
- Observability integrations

✅ **Only change needed**:

- Add `"use_litellm": true` to model configuration

## Related Documentation

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Buttermilk LLM Architecture](./bots/_CHUNKS/LLM_ARCHITECTURE.md)
- [Issue #289: LiteLLM Migration](https://github.com/qut-dmrc/buttermilk/issues/289)

## Support

For issues or questions:

1. Check [Issue #289](https://github.com/qut-dmrc/buttermilk/issues/289) for latest updates
1. Review LiteLLM documentation for provider-specific details
1. Open a new issue if you encounter problems during migration
