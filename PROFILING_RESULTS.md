# Buttermilk Initialization Profiling Results

**Date:** 2025-11-03 **Total Cold Start Time:** ~14.7 seconds **Import Time:** ~14.5 seconds (98.6%) **Actual Init Time:** ~0.2 seconds (1.4%)

## 🔴 Critical Finding

**90%+ of startup time is spent on imports, not configuration or initialization.**

Your initialization code is already well-optimized. The bottleneck is **eager module-level imports** that load heavy dependencies even when they're not needed.

## 📊 Import Time Breakdown (CLI mode)

| Package           | Time      | % of Total | Status                           |
| ----------------- | --------- | ---------- | -------------------------------- |
| **wandb**         | **64.2s** | **35.4%**  | 🔴 Via weave, not directly used  |
| **weave**         | **27.0s** | **14.9%**  | 🔴 Imported eagerly in \_core    |
| **google.**\*     | **22.4s** | **12.3%**  | 🟡 GCP clients (aiplatform, etc) |
| **litellm**       | **5.8s**  | **3.2%**   | 🟡 Via dependencies              |
| **opentelemetry** | **3.6s**  | **2.0%**   | 🟡 Observability                 |
| **vertexai**      | **3.1s**  | **1.7%**   | 🟡 GCP AI Platform               |
| openai            | 2.1s      | 1.1%       | Used for LLMs                    |
| anthropic         | 0.99s     | 0.5%       | Used for LLMs                    |
| chromadb          | 0.44s     | 0.2%       | Vector DB                        |
| fastapi           | 0.48s     | 0.3%       | API mode only                    |

**Total cumulative import time:** 181.5 seconds (with transitive dependencies) **Actual wall clock time:** ~14.5 seconds

## 🎯 Root Cause Analysis

### 1. Weave Import (PRIMARY ISSUE)

**File:** `buttermilk/_core/bm_init.py:38`

```python
import weave  # For tracing - core dependency
```

**Impact:**

- Directly adds **27 seconds** of cumulative import time
- Transitively imports **wandb** (64 seconds cumulative)
- Combined: **91 seconds** (50% of all import time)
- Wall clock: ~6-8 seconds

**Why it's a problem:**

- Imported at module level in `_core/bm_init.py`
- This means EVERY buttermilk import triggers weave/wandb loading
- Weave/wandb are only needed if tracing is enabled

**Files with eager weave imports:**

```
buttermilk/_core/bm_init.py:38
buttermilk/_core/tracing.py:3
buttermilk/_core/execution_context.py:26
buttermilk/_core/llm_core.py:17
buttermilk/_core/llms.py:21
buttermilk/_core/orchestrator.py:25
buttermilk/_core/agent.py:19
buttermilk/_core/standalone_trace.py:14
buttermilk/pipeline.py:63
```

### 2. Google Cloud Platform Imports (SECONDARY ISSUE)

**Impact:** 22.4 seconds cumulative (12.3%)

**Why it's a problem:**

- GCP client libraries (aiplatform, vertex AI) are imported even if not configured
- Should only import when `infrastructure.clouds` contains GCP config
- Likely imported in `execution_context.py` during infrastructure setup

### 3. LiteLLM & OpenTelemetry

**Impact:** Combined ~9 seconds

These are likely transitive dependencies from weave or other libraries, not direct imports.

## 🚀 Optimization Strategy

### Phase 1: Lazy Load Weave (HIGHEST IMPACT)

**Expected improvement:** 6-8 seconds faster startup (~50% reduction)

**Implementation:**

1. **Make weave imports conditional in `_core/bm_init.py`:**

```python
# BEFORE (line 38):
import weave  # For tracing - core dependency

# AFTER:
# Lazy import weave only when tracing is needed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import weave  # For type checking only

def _get_weave_client():
    """Lazy import weave only when needed."""
    import weave
    return weave.client()
```

2. **Update all weave usage to be lazy:**

In files that use weave decorators (`@weave.op`), you need a different approach since decorators are evaluated at import time:

```python
# Option A: Conditional decorator
def conditional_weave_op(func):
    """Apply @weave.op only if tracing is enabled."""
    if os.getenv('WEAVE_ENABLED') == 'true':
        import weave
        return weave.op(func)
    return func

@conditional_weave_op
def my_function():
    pass
```

Or:

```python
# Option B: Defer decorator application
def my_function():
    pass

# At module end or in init function:
if tracing_enabled:
    import weave
    my_function = weave.op(my_function)
```

3. **Update tracing initialization:**

In `_core/tracing.py`, only import weave when `init_weave()` is actually called:

```python
# _core/tracing.py
def init_weave(project_name: str, enabled: bool = True):
    """Initialize weave tracing - imports weave only if enabled."""
    if not enabled:
        logger.info("Weave tracing disabled")
        return None

    # Lazy import here
    import weave

    weave.init(project_name)
    return weave.client()
```

### Phase 2: Lazy Load Cloud Clients (MEDIUM IMPACT)

**Expected improvement:** 1-2 seconds

**Implementation:**

In `_core/execution_context.py`, defer GCP/Azure client imports until actually needed:

```python
# BEFORE:
from google.cloud import aiplatform
from google.cloud import bigquery

# AFTER:
def _get_gcp_client(service: str):
    """Lazy import GCP clients."""
    if service == 'aiplatform':
        from google.cloud import aiplatform
        return aiplatform
    elif service == 'bigquery':
        from google.cloud import bigquery
        return bigquery
```

### Phase 3: Optional Optimizations (LOW IMPACT)

1. **ChromaDB** - Already only 0.44s, but could be lazy loaded in storage modules
1. **FastAPI** - Should already only load in API mode (verify not in runner.cli)
1. **Pandas/Numpy** - Standard data libs, acceptable overhead

## 📈 Expected Results

| Optimization     | Current | After   | Improvement |
| ---------------- | ------- | ------- | ----------- |
| **Baseline**     | 14.7s   | -       | -           |
| **+ Lazy Weave** | 14.7s   | **~7s** | **-52%**    |
| **+ Lazy Cloud** | ~7s     | **~5s** | **-66%**    |
| **Target**       | 14.7s   | **~5s** | **-66%**    |

## 🔧 Implementation Priority

### Priority 1: Weave (CRITICAL)

- [ ] Make weave import conditional in `_core/bm_init.py`
- [ ] Update `_core/tracing.py` to lazy load
- [ ] Handle `@weave.op` decorators conditionally
- [ ] Update all other `_core` files with weave imports
- [ ] Add config flag: `observability.weave.enabled` (default: true)

### Priority 2: Cloud Clients (HIGH)

- [ ] Lazy load GCP clients in `execution_context.py`
- [ ] Only import when cloud is configured
- [ ] Add early returns for unconfigured clouds

### Priority 3: Verification (MEDIUM)

- [ ] Verify FastAPI not imported in non-API modes
- [ ] Check for other eager imports in `_core`
- [ ] Profile again to verify improvements

## 🧪 Testing Strategy

After implementing lazy loading:

```bash
# 1. Profile again
uv run python scripts/profile_init.py

# 2. Verify functionality with tracing OFF
uv run python -m buttermilk.runner.cli run.mode=console run.flow=trans observability.weave.enabled=false

# 3. Verify functionality with tracing ON (should still work)
uv run python -m buttermilk.runner.cli run.mode=console run.flow=trans observability.weave.enabled=true

# 4. Compare import times
uv run python -X importtime -c "from buttermilk._core import config_bootstrap" 2> after_optimization.txt
```

## 📝 Notes

1. **Weave decorators** are tricky because they're evaluated at import time. You may need to:

   - Use a conditional decorator factory
   - Apply decorators dynamically after import
   - Or use environment variable to control decorator behavior

1. **Type checking** - Use `TYPE_CHECKING` for type hints without importing:

   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       import weave
   ```

1. **Backwards compatibility** - Consider adding a flag to enable eager imports for users who want the old behavior:

   ```yaml
   # config.yaml
   performance:
     lazy_imports: true  # Default true for speed
   ```

## 🎯 Quick Wins Summary

The **single biggest win** is making weave imports lazy. This alone will cut startup time in half.

The issue isn't your initialization logic (which is already fast at ~200ms). It's that you're importing heavyweight observability libraries that bring in 90+ seconds of cumulative dependency chains, even when users might not need tracing.

**Next Steps:**

1. Implement lazy weave loading
1. Profile again to verify improvement
1. Iterate on cloud client lazy loading if needed
