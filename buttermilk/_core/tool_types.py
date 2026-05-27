"""Native tool protocol types.

Provides:
- Tool: Protocol for executable tools with schema
- FunctionTool: Wraps a Python callable into a Tool
- ToolSchema: JSON Schema wrapper for tool parameters
- CancellationToken: Simple cooperative cancellation
"""

import asyncio
import functools
import inspect
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import TypeAdapter

from buttermilk._core.messages import FunctionCall


@dataclass
class ToolSchema:
    """JSON Schema description of a tool's interface."""

    name: str
    description: str
    parameters: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools that can be invoked by LLMs."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def schema(self) -> ToolSchema: ...

    async def run_json(self, args: dict[str, Any], cancellation_token: "CancellationToken") -> Any: ...

    def return_value_as_string(self, value: Any) -> str: ...


class CancellationToken:
    """Cooperative cancellation token.

    Allows callers to signal cancellation to async operations.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError("Operation was cancelled")


def _generate_schema(func: Any, name: str, description: str, strict: bool) -> dict[str, Any]:
    """Generate JSON Schema from a callable's type hints."""
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = func.__annotations__ if hasattr(func, "__annotations__") else {}
    except Exception:
        pass

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "return"):
            continue

        annotation = hints.get(param_name, Any)
        if annotation is inspect.Parameter.empty:
            annotation = Any

        try:
            adapter = TypeAdapter(annotation)
            schema = adapter.json_schema()
        except Exception:
            schema = {"type": "string"}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        elif not strict:
            schema["default"] = param.default if param.default is not inspect.Parameter.empty else None

        properties[param_name] = schema

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required

    if strict:
        result["additionalProperties"] = False

    return result


class FunctionTool:
    """Wraps a Python callable as a Tool with auto-generated JSON Schema.

    Args:
        func: The callable to wrap (sync or async).
        name: Tool name for LLM identification.
        description: Human-readable description of the tool.
        strict: If True, disallow additional properties in schema.
    """

    def __init__(
        self,
        func: Any,
        *,
        name: str,
        description: str,
        strict: bool = False,
    ) -> None:
        self._func = func
        self._name = name
        self._description = description
        self._strict = strict
        self._parameters = _generate_schema(func, name, description, strict)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
        )

    async def run_json(self, args: dict[str, Any], cancellation_token: CancellationToken | None = None) -> Any:
        if cancellation_token and cancellation_token.is_cancelled:
            raise asyncio.CancelledError("Operation was cancelled")

        if asyncio.iscoroutinefunction(self._func):
            return await self._func(**args)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self._func, **args))

    def return_value_as_string(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)

    def args_type(self) -> type:
        return dict

    def return_type(self) -> type:
        return dict

    def state_type(self) -> type:
        return type(None)

    async def save_state_json(self) -> str:
        return "{}"

    async def load_state_json(self, state_json: str) -> None:
        pass
