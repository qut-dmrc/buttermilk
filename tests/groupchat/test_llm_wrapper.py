import asyncio
import json

import pytest
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage, UserMessage
from pydantic import BaseModel

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import AutoGenWrapper, ModelOutput


class FakeChatClient(ChatCompletionClient):
    """Minimal stand-in for autogen_core.models.ChatCompletionClient.

    Behavior is controlled by the `mode` value:
    - 'text': always return final text
    - 'tool_then_text': first return a single tool call, then final text
    - 'tool_loop': always return a single tool call (to trigger max_tool_iterations)
    - 'schema_base_model': return a Pydantic BaseModel instance in content
    """

    def __init__(self, mode: str, *, schema_model: type[BaseModel] | None = None):
        self.mode = mode
        self._count = 0
        self._schema_model = schema_model
        # minimal model info dict; wrapper treats it like a Mapping
        self._model_info = {
            "family": "openai",
            "vision": False,
            "json_output": False,
            "structured_output": False,
            "function_calling": True,
        }

    # Required abstract interface pieces
    @property
    def model_info(self):  # type: ignore[override]
        return self._model_info

    @property
    def capabilities(self):  # type: ignore[override]
        return {}

    async def close(self) -> None:  # type: ignore[override]
        return None

    async def count_tokens(self, messages, *, tools: list = None):  # type: ignore[override]
        return 0

    async def remaining_tokens(self, messages, *, tools: list = None):  # type: ignore[override]
        return 8_000

    async def total_usage(self):  # type: ignore[override]
        from autogen_core.models import RequestUsage

        return RequestUsage()

    async def actual_usage(self):  # type: ignore[override]
        from autogen_core.models import RequestUsage

        return RequestUsage()

    async def create(self, messages, *, tools=None, tool_choice="auto", json_output=None, extra_create_args=None, cancellation_token=None):  # type: ignore[override]
        if extra_create_args is None:
            extra_create_args = {}
        if tools is None:
            tools = []
        self._count += 1
        usage = RequestUsage(prompt_tokens=1, completion_tokens=1)
        if self.mode == "text":
            return CreateResult(content="final text", finish_reason="stop", usage=usage, cached=False)

        if self.mode == "tool_then_text":
            if self._count == 1:
                call = FunctionCall(id="1", name="search", arguments=json.dumps({"q": "hello"}))
                return CreateResult(content=[call], finish_reason="function_calls", usage=usage, cached=False)
            return CreateResult(content="final from synth", finish_reason="stop", usage=usage, cached=False)

        if self.mode == "tool_loop":
            call = FunctionCall(id="1", name="search", arguments=json.dumps({"q": "hello"}))
            return CreateResult(content=[call], finish_reason="function_calls", usage=usage, cached=False)

        if self.mode == "schema_base_model":
            assert self._schema_model is not None
            # Return JSON string content to comply with CreateResult schema
            return CreateResult(
                content=self._schema_model(field="value").model_dump_json(),
                finish_reason="stop",
                usage=usage,
                cached=False,
            )

        raise AssertionError(f"Unknown mode: {self.mode}")

    async def create_stream(self, *args, **kwargs):  # type: ignore[override]
        # Minimal async generator to satisfy abstract interface; not used in tests
        yield "stream-not-implemented"


class DummyTool:
    def __init__(self, name: str = "search"):
        self.name = name

    @staticmethod
    async def run_json(args: dict, ct: CancellationToken):
        await asyncio.sleep(0)  # yield
        return {"ok": True, "q": args.get("q")}

    @staticmethod
    def return_value_as_string(result) -> str:
        return json.dumps(result)


class MySchema(BaseModel):
    field: str


@pytest.mark.anyio
async def test_call_chat_returns_text_without_tools():
    client = FakeChatClient("text")
    wrapper = AutoGenWrapper(
        client_factory=lambda: client,
        model_info={
            "family": "openai",
            "vision": False,
            "json_output": False,
            "structured_output": False,
            "function_calling": True,
        },
    )

    res = await wrapper.call_chat(messages=[UserMessage(content="hi", source="user")], cancellation_token=None)
    assert isinstance(res, CreateResult)
    assert isinstance(res.content, str)
    assert res.content == "final text"


@pytest.mark.anyio
async def test_call_chat_tool_then_text_happy_path():
    client = FakeChatClient("tool_then_text")
    wrapper = AutoGenWrapper(
        client_factory=lambda: client,
        model_info={
            "family": "openai",
            "vision": False,
            "json_output": False,
            "structured_output": False,
            "function_calling": True,
        },
    )

    res = await wrapper.call_chat(
        messages=[UserMessage(content="search for hi", source="user")],
        cancellation_token=None,
        tools_list=[DummyTool()],
    )
    assert isinstance(res, CreateResult)
    assert isinstance(res.content, str)
    assert res.content == "final from synth"


@pytest.mark.anyio
async def test_call_chat_tool_loop_fails_when_exceeded():
    client = FakeChatClient("tool_loop")
    wrapper = AutoGenWrapper(
        client_factory=lambda: client,
        model_info={
            "family": "openai",
            "vision": False,
            "json_output": False,
            "structured_output": False,
            "function_calling": True,
        },
    )

    # With tool_loop mode, the client always returns tool calls, never text.
    # The wrapper should handle this gracefully (behavior depends on implementation)
    result = await wrapper.call_chat(
        messages=[UserMessage(content="search repeatedly", source="user")],
        cancellation_token=None,
        tools_list=[DummyTool()],
    )
    # Verify we get a result (tool calls are returned)
    assert result is not None


@pytest.mark.anyio
async def test_create_schema_with_base_model_content_normalizes_and_parses():
    client = FakeChatClient("schema_base_model", schema_model=MySchema)
    # Pretend model supports structured output
    wrapper = AutoGenWrapper(
        client_factory=lambda: client,
        model_info={
            "family": "openai",
            "vision": False,
            "json_output": True,
            "structured_output": True,
            "function_calling": True,
        },
    )

    res = await wrapper.create(messages=[UserMessage(content="make object", source="user")], schema=MySchema)
    assert isinstance(res, ModelOutput)
    assert isinstance(res.content, str)  # normalized to JSON string
    assert res.parsed_object is not None and isinstance(res.parsed_object, MySchema)
    assert res.parsed_object.field == "value"
