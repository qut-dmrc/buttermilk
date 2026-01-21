# Processor Architecture Unification

**Status:** Approved
**Author:** Claude (via design review)
**Date:** 2025-12-15
**Related PRs:** #303, branch `claude/unify-processors-traces-XaNSR`
**Constraint:** NO backwards compatibility. Clean break.

## Problem Statement

Buttermilk has three processor base classes that share significant infrastructure (tracing, trace_writer, metadata building) but have incompatible initialization patterns:

| Class | Inherits From | Initialization | Used By |
|-------|--------------|----------------|---------|
| `LLMCore` | `ProcessorCore` | `__init__(**kwargs)` | LLM-based processors |
| `ClassifierCore` | `ProcessorCore` | `__init__(**kwargs)` | Classification APIs |
| `ToxicityClassifierCore` | `BaseModel` | Pydantic fields | Toxicity/safety APIs |

This creates:
1. **Code duplication** - `trace_writer` property and trace emission logic repeated
2. **Inconsistent interfaces** - Different ways to access `parameters`
3. **Maintenance burden** - Changes must be synchronized across classes
4. **Type confusion** - Some processors are Pydantic models, others aren't

## Current State

```
ProcessorCore (plain class)
├── __init__(self, **kwargs)
│   └── self.parameters = kwargs
│   └── self._trace_writer = None
├── trace_writer property
├── _emit_success_trace()
├── _emit_error_trace()
└── abstract process()

LLMCore(ProcessorCore)
└── __init__(model, template, output_model, ...)

ClassifierCore(ProcessorCore)
└── __init__(template, output_model, ...)

ToxicityClassifierCore(BaseModel)  # SEPARATE HIERARCHY
├── model, process_chain, standard (Pydantic fields)
├── _trace_writer (duplicated)
├── trace_writer property (duplicated)
└── _emit_trace_for_moderation() (custom trace logic)
```

## Design Goals

1. **Single source of truth** for tracing infrastructure
2. **Consistent initialization** across all processor types
3. **Type safety** via Pydantic validation where beneficial
4. **Minimal breaking changes** to existing code
5. **Clear extension patterns** for new processor types

---

## Approved Design: Unified Pydantic BaseModel Hierarchy

Convert `ProcessorCore` to inherit from `BaseModel`, making all processors Pydantic models.

### Architecture

```
ProcessorCore(BaseModel)
├── parameters: dict[str, Any] = Field(default_factory=dict)
├── _trace_writer: Any = PrivateAttr(default=None)
├── trace_writer property
├── _emit_success_trace()
├── _emit_error_trace()
├── model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
└── abstract process()

LLMCore(ProcessorCore)
├── model: str
├── template: str
├── output_model: type[BaseModel] | str | None = None
├── tools: list[Tool] = Field(default_factory=list)
└── process()

ClassifierCore(ProcessorCore)
├── template: str
├── output_model: type[BaseModel] | str
└── process()

ToxicityClassifierCore(ProcessorCore)
├── model: str
├── process_chain: str
├── standard: str
├── client: Any = None
└── process()
```

### Key Design Decisions

**1. Use `PrivateAttr` for mutable state**
```python
from pydantic import PrivateAttr

class ProcessorCore(BaseModel):
    _trace_writer: Any = PrivateAttr(default=None)
```

**2. Compute `parameters` from model fields**
```python
@computed_field
@property
def parameters(self) -> dict[str, Any]:
    """Return all public fields as parameters dict for tracing."""
    return {
        k: v for k, v in self.model_dump().items()
        if not k.startswith('_') and k not in ('client',)
    }
```

**3. Use `model_validator` for initialization logic**
```python
@model_validator(mode="after")
def _init_processor(self) -> Self:
    """Post-initialization: resolve string paths, validate config."""
    if isinstance(self.output_model, str):
        self.output_model = import_class_from_path(self.output_model)
    return self
```

**4. Abstract method via ABC + BaseModel**
```python
from abc import ABC, abstractmethod

class ProcessorCore(ABC, BaseModel):
    @abstractmethod
    async def process(self, record: BaseRecord, ...) -> AsyncGenerator[BaseRecord, None]:
        ...
```

### Migration Path

**Phase 1: Convert ProcessorCore**
```python
# Before
class ProcessorCore:
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self._trace_writer = None

# After
class ProcessorCore(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    _trace_writer: Any = PrivateAttr(default=None)

    @computed_field
    @property
    def parameters(self) -> dict[str, Any]:
        return self.model_dump(exclude={'client'})
```

**Phase 2: Convert LLMCore**
```python
# Before
class LLMCore(ProcessorCore):
    def __init__(self, model: str, template: str, ...):
        super().__init__(model=model, template=template, ...)
        self.model = model
        self.template = template

# After
class LLMCore(ProcessorCore):
    model: str
    template: str
    output_model: type[BaseModel] | str | None = None
    tools: list[Tool] = Field(default_factory=list)
    fail_on_unfilled_parameters: bool = True
    output_col: str = "output"

    # Template metadata tracked as private attr
    _template_metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
```

**Phase 3: Convert ClassifierCore**
```python
class ClassifierCore(ProcessorCore):
    template: str
    output_model: type[BaseModel] | str
    output_col: str = "output"

    @model_validator(mode="after")
    def _resolve_output_model(self) -> Self:
        if isinstance(self.output_model, str):
            self.output_model = import_class_from_path(self.output_model)
        return self
```

**Phase 4: Rebase ToxicityClassifierCore**
```python
# Before: ToxicityClassifierCore(BaseModel) with duplicated trace logic
# After: ToxicityClassifierCore(ProcessorCore) inheriting everything

class ToxicityClassifierCore(ProcessorCore):
    model: str
    process_chain: str
    standard: str
    client: Any = None
    info_url: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    options: ClassVar[dict] = {}
    call_options: ClassVar[dict] = {}

    @model_validator(mode="after")
    def _init_client(self) -> Self:
        if self.client is None:
            self.init_client(**self.options)
        return self
```

### Pros
- **Single unified hierarchy** - all processors are Pydantic models
- **Consistent initialization** - field declarations, not `__init__` parameters
- **Automatic validation** - Pydantic validates all inputs
- **Serialization** - `model_dump()`, `model_json_schema()` for free
- **Clear extension pattern** - new processors inherit from `ProcessorCore`
- **No shims** - `parameters` computed uniformly from fields

### Cons
- **Breaking change** for existing instantiation code
- **Migration effort** - must update all processor instantiations
- **Learning curve** - developers must understand Pydantic patterns

### Breaking Changes

Instantiation syntax changes:
```python
# Before
llm = LLMCore(model="gpt-4", template="classify", output_model=MyModel)

# After (identical - Pydantic accepts kwargs!)
llm = LLMCore(model="gpt-4", template="classify", output_model=MyModel)
```

Actually, **instantiation is identical** because Pydantic models accept kwargs. The main breaking changes are:

1. Subclasses that override `__init__` must migrate to `model_validator`
2. Direct attribute assignment after construction may fail (Pydantic frozen/validation)
3. `isinstance(processor, BaseModel)` will now be `True` for all processors

### Implementation Complexity
**Medium** - Requires touching all processor classes, but changes are mechanical

---

## Option C: Protocol-Based Structural Typing

Define `Processor` as a Protocol, use composition for shared behavior.

### Architecture

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Processor(Protocol):
    """Structural interface for all processors."""

    @property
    def parameters(self) -> dict[str, Any]: ...

    @property
    def trace_writer(self) -> Any: ...

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[BaseRecord, None]: ...

class TracingMixin:
    """Shared tracing implementation."""
    # Same as Option A

class LLMCore(TracingMixin):  # No base class
    def __init__(self, model: str, template: str, ...): ...

class ToxicityClassifierCore(TracingMixin, BaseModel):
    # Pydantic model with mixin
```

### Pros
- Maximum flexibility - no inheritance constraints
- Duck typing - anything implementing the protocol works
- Gradual adoption - can migrate incrementally

### Cons
- No shared implementation without mixin (back to Option A)
- Protocol checking is runtime, not static without plugins
- Developers must manually ensure protocol compliance
- Doesn't solve the two-hierarchy problem

### Implementation Complexity
**Low-Medium** - Protocol is simple, but mixin still needed

---

## Appendix: Code Sketches

### ProcessorCore

```python
"""Shared base class for all pipeline processors."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field

from buttermilk import logger
from buttermilk._core.types import BaseRecord


class ProcessorCore(ABC, BaseModel):
    """Base class for all pipeline processors.

    Provides:
    - Lazy trace_writer property for BigQuery persistence
    - Standardized trace emission (_emit_success_trace, _emit_error_trace)
    - Consistent parameters dict for observability

    Subclasses define their configuration as Pydantic fields and implement
    the abstract process() method.

    Example:
        class MyProcessor(ProcessorCore):
            model: str
            threshold: float = 0.5

            async def process(self, record, *, processor_stage, **kwargs):
                result = await self._do_work(record)
                yield record.model_copy(update={"output": result})
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # Private attributes (not included in serialization)
    _trace_writer: Any = PrivateAttr(default=None)

    @computed_field
    @property
    def parameters(self) -> dict[str, Any]:
        """Return processor configuration for tracing.

        Excludes private attributes and non-serializable fields like 'client'.
        """
        excluded = {'client', 'credentials'}
        return {
            k: v for k, v in self.model_dump().items()
            if k not in excluded
        }

    @property
    def trace_writer(self) -> Any:
        """Lazy-load trace writer for BigQuery persistence."""
        if self._trace_writer is None:
            try:
                from buttermilk.utils.trace_writer import get_trace_writer
                self._trace_writer = get_trace_writer()
            except Exception as e:
                logger.warning(f"Failed to initialize trace writer: {e}")
        return self._trace_writer

    async def _emit_trace(self, execution_trace: "ExecutionTrace") -> None:
        """Emit execution trace with error handling."""
        if self.trace_writer:
            try:
                await self.trace_writer.add(execution_trace)
            except Exception as e:
                logger.warning(f"Failed to emit trace: {e}")

    def _build_agent_info(
        self,
        processor_stage: str,
        execution_type: str = "processing",
    ) -> dict[str, Any]:
        """Build standardized agent_info dict for ExecutionTrace."""
        return {
            "component_name": self.__class__.__name__,
            "execution_type": execution_type,
            "processor_stage": processor_stage,
            "config": self.parameters,
        }

    async def _emit_success_trace(
        self,
        record: BaseRecord,
        outputs: Any,
        processor_stage: str,
        parent_trace_id: str | None,
        duration_ms: float,
        **kwargs,
    ) -> str:
        """Emit a success execution trace. Returns trace_id."""
        import uuid
        from buttermilk._core.contract import ExecutionTrace

        trace_id = str(uuid.uuid4())
        execution_trace = ExecutionTrace(
            call_id=trace_id,
            agent_info=self._build_agent_info(processor_stage),
            outputs=outputs,
            parameters=self.parameters,
            metadata={"duration_ms": duration_ms, **kwargs.get("extra_metadata", {})},
            parent_call_id=parent_trace_id,
            record=record,
            **{k: v for k, v in kwargs.items() if k != "extra_metadata"},
        )
        await self._emit_trace(execution_trace)
        return trace_id

    async def _emit_error_trace(
        self,
        record: BaseRecord | None,
        error: Exception,
        processor_stage: str,
        parent_trace_id: str | None,
        duration_ms: float,
        **kwargs,
    ) -> None:
        """Emit an error execution trace."""
        from buttermilk._core.contract import ExecutionTrace

        error_trace = ExecutionTrace(
            agent_info=self._build_agent_info(processor_stage),
            error={"event": str(error), "details": {"error_type": type(error).__name__}},
            parameters=self.parameters,
            metadata={"duration_ms": duration_ms},
            parent_call_id=parent_trace_id,
            record=record,
        )
        await self._emit_trace(error_trace)

    @abstractmethod
    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord and yield enriched output records.

        Subclasses must implement this method.
        """
        raise NotImplementedError
        yield  # Make this a generator
```

### LLMCore

```python
class LLMCore(ProcessorCore):
    """Core LLM functionality for template-based LLM processing."""

    model: str
    template: str
    output_model: str | None = None  # Keep as string for serialization
    tools: list[Tool] = Field(default_factory=list)
    fail_on_unfilled_parameters: bool = True
    output_col: str = "output"

    # Private state
    _template_metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    _resolved_output_model: type[BaseModel] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve_output_model(self) -> Self:
        """Resolve string output_model paths to classes."""
        if isinstance(self.output_model, str):
            self._resolved_output_model = import_class_from_path(self.output_model)
        else:
            self._resolved_output_model = self.output_model
        return self

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[BaseRecord, None]:
        # Implementation using self.model, self.template, etc.
        ...
```

---

## Review Feedback (2025-12-15)

### Problems & Risks

1.  **Serialization of `output_model` (Critical)**
    The proposal suggests resolving `output_model` from a `str` to a class (`type[BaseModel]`) either in-place (Phase 3 text) or into a private attribute (LLMCore sketch).
    *   **The Issue:** If `output_model` is mutated to be a class object, `model_dump()` (used by `parameters`) will return a dictionary containing a Python class object. This is **not JSON serializable**, which will likely cause the trace writer (BigQuery/JSON) to crash when trying to log `parameters`.
    *   **Inconsistency:** The text for Phase 3 suggests mutating `self.output_model` directly, while the `LLMCore` code sketch uses `_resolved_output_model`.

2.  **`client` Field Overhead in `ToxicityClassifierCore`**
    The proposal defines `client: Any = None` as a public field.
    *   **The Issue:** Even though it is manually excluded from the `parameters` property, it is still part of the Pydantic model's standard fields. This can cause issues with `model_dump()` in other contexts (e.g., debugging) and adds overhead if Pydantic tries to validate or copy heavy client objects (like Google Cloud clients) during model operations.

3.  **Validation vs. Logic in `model_validator`**
    Using `model_validator(mode="after")` to perform heavy imports (`import_class_from_path`) or side-effects (initializing clients) effectively puts "constructor logic" into validation. If these imports fail or are slow, simply instantiating the class for inspection or testing becomes heavy/risky.

### Recommended Improvements

1.  **Separate Configuration from Runtime State (Strongly Recommended)**
    Do not mutate configuration fields. Keep `output_model` as the string/config representation and store the resolved class in a `PrivateAttr`.
    *   **Why:** Ensures `model_dump()` always returns clean, JSON-serializable configuration data for tracing.
    *   **Fix:**
        ```python
        class LLMCore(ProcessorCore):
            output_model: str | None = None  # Keep as config (str)
            _resolved_output_model: type[BaseModel] | None = PrivateAttr(default=None)

            @model_validator(mode="after")
            def _resolve_model(self) -> Self:
                if self.output_model:
                     self._resolved_output_model = import_class_from_path(self.output_model)
                return self
        ```

2.  **Mark `client` as `PrivateAttr`**
    Explicitly mark the client and other runtime-only objects as private attributes in `ToxicityClassifierCore`.
    *   **Fix:** `_client: Any = PrivateAttr(default=None)` instead of `client: Any = None`. This automatically excludes it from `model_dump()` and validation.

3.  **Standardize `output_model` Type Hint**
    In `LLMCore`, the type is `type[BaseModel] | str | None`. If you adopt the "separate config" recommendation, the public field should be stricter (just `str` if it's always loaded from config, or `str | None`). If you allow passing a class directly in `__init__`, you need a custom validator that extracts the class name for the public field (for serialization) and stores the class reference in the private attribute.

4.  **Trace Writer Safety**
    The lazy loading of `trace_writer` inside the property imports from `buttermilk.utils.trace_writer`. Ensure this import is wrapped in a try/except `ImportError` block specifically to avoid crashing if the environment is minimal, not just catching generic `Exception`.