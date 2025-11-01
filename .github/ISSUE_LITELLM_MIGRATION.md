# Migrate from Autogen to LiteLLM for LLM API calls

## Overview

Migrate from Autogen to LiteLLM for all LLM API calls while maintaining backward compatibility with existing ExecutionTrace and observability infrastructure.

## Current State

**Dependencies:**
- **Autogen** (`autogen-core`, `autogen-ext`): Currently handles all LLM API calls via `ChatCompletionClient`
- **LiteLLM**: Only used for token cost calculation (`buttermilk/utils/pricing.py`)

**Key Components:**
1. `buttermilk/_core/llms.py`: Manages LLM configurations and clients
   - `LLMConfig`: Stores provider config
   - `AutoGenWrapper`: Wraps Autogen's ChatCompletionClient
   - `LLMs`: Registry for LLM clients

2. `buttermilk/_core/llm_core.py`: Core LLM operations
   - `LLMCore`: Template rendering, calling, tracing
   - `LLMResult`: Result structure with metadata

3. `buttermilk/agents/llm.py`: Agent wrapper around LLMCore

**Current Providers:**
- OpenAI (direct and via Azure)
- Anthropic (direct and via Vertex)
- Google Gemini (via Vertex)
- Vertex OpenAI endpoints (for Llama, etc.)

## Migration Strategy

### Phase 1: Create LiteLLM Wrapper
**Goal:** Build a drop-in replacement for `AutoGenWrapper` using LiteLLM

**Tasks:**
- [ ] Create `LiteLLMWrapper` class in `buttermilk/_core/llms.py`
  - [ ] Implement `.create()` method matching `AutoGenWrapper` interface
  - [ ] Implement `.call_chat()` for tool execution flow
  - [ ] Convert between Autogen message types and LiteLLM format
  - [ ] Preserve retry logic (copy from `RetryWrapper`)
  - [ ] Maintain structured output parsing
  - [ ] Support tool calling/function execution
  - [ ] Integrate pricing calculation (already using LiteLLM)

- [ ] Message format conversion utilities
  - [ ] `autogen_to_litellm_messages()`: Convert LLMMessage → LiteLLM format
  - [ ] `litellm_to_autogen_result()`: Convert LiteLLM response → CreateResult/ModelOutput
  - [ ] Handle tool calls in both directions

**Acceptance Criteria:**
- `LiteLLMWrapper` implements same interface as `AutoGenWrapper`
- All message types convert correctly
- Structured output parsing works identically
- Tool calling flow matches existing behavior

### Phase 2: Update Configuration
**Goal:** Allow per-model selection of wrapper type

**Tasks:**
- [ ] Add `use_litellm: bool` flag to `LLMConfig`
- [ ] Update `LLMs.get_autogen_chat_client()` to:
  - [ ] Check `use_litellm` flag
  - [ ] Return `LiteLLMWrapper` when flag is True
  - [ ] Return `AutoGenWrapper` (existing) when flag is False
  - [ ] Default to False for backward compatibility

- [ ] Simplify provider configuration for LiteLLM
  - [ ] Map `ClientType` enum to LiteLLM provider names
  - [ ] Leverage LiteLLM's built-in provider support
  - [ ] Reduce custom credential handling where possible

**Acceptance Criteria:**
- Can toggle wrapper type per model via config
- Both wrapper types work side-by-side
- No breaking changes to existing configs

### Phase 3: Testing & Validation
**Goal:** Ensure full compatibility and correctness

**Tasks:**
- [ ] Unit tests for `LiteLLMWrapper`
  - [ ] Test basic completion calls
  - [ ] Test structured output with schemas
  - [ ] Test tool calling flow
  - [ ] Test retry logic
  - [ ] Test error handling
  - [ ] Test pricing calculation

- [ ] Integration tests
  - [ ] Run existing LLM tests with both wrapper types
  - [ ] Verify ExecutionTrace metadata matches
  - [ ] Verify Weave tracing works correctly
  - [ ] Test all supported providers (OpenAI, Azure, Anthropic, Gemini, Vertex)

- [ ] End-to-end tests
  - [ ] Run existing flows with LiteLLM wrapper
  - [ ] Verify complete observability chain
  - [ ] Verify agent behavior unchanged

**Acceptance Criteria:**
- All existing tests pass with LiteLLM wrapper
- ExecutionTrace metadata identical between wrappers
- No observable behavioral differences

### Phase 4: Gradual Migration
**Goal:** Migrate models one at a time

**Tasks:**
- [ ] Update `conf/models.json` to enable LiteLLM for specific models
- [ ] Start with one low-risk model (e.g., `gemini25flash`)
- [ ] Monitor production usage and observability
- [ ] Gradually enable for more models
- [ ] Document any provider-specific quirks

**Acceptance Criteria:**
- At least one production model using LiteLLM
- Observability and tracing working correctly
- No production incidents

### Phase 5: Deprecation (Future)
**Goal:** Eventually remove Autogen dependency

**Tasks:**
- [ ] Migrate all models to LiteLLM
- [ ] Remove `AutoGenWrapper` class
- [ ] Remove Autogen dependencies from `pyproject.toml`
- [ ] Update documentation
- [ ] Keep Autogen message types (widely used in codebase)

**Note:** This phase is optional and can be deferred indefinitely if both systems work well.

## Benefits

✅ **Provider flexibility**: LiteLLM supports 100+ providers natively
✅ **Simplified configuration**: Uniform interface across all providers
✅ **Better cost tracking**: Already using LiteLLM for pricing
✅ **Gradual migration**: Test per-model without breaking existing flows
✅ **Zero breaking changes**: Existing API stays fully intact
✅ **Reduced custom code**: Less provider-specific credential handling
✅ **Future-proof**: Easy to add new providers as they emerge

## Non-Goals

❌ Changing `ExecutionTrace` structure
❌ Modifying `LLMCore` or `LLMAgent` interfaces
❌ Breaking backward compatibility
❌ Removing Autogen message types (used throughout codebase)

## Success Criteria

- [ ] LiteLLM wrapper implemented with full feature parity
- [ ] All tests pass with both wrapper types
- [ ] At least one production model using LiteLLM successfully
- [ ] Observability and tracing fully functional
- [ ] Zero breaking changes to existing code
- [ ] Documentation updated

## Related Issues

None yet

## Timeline

- **Phase 1:** 2-3 days (wrapper implementation)
- **Phase 2:** 1 day (configuration updates)
- **Phase 3:** 2-3 days (comprehensive testing)
- **Phase 4:** Ongoing (gradual rollout)
- **Phase 5:** TBD (optional deprecation)

**Total estimated time:** 1-2 weeks for initial implementation and testing

## Labels

`enhancement`, `infrastructure`, `llm`, `migration`
