"""Tests for reasoning-model handling in litellm_to_autogen_result.

Covers three response shapes we encounter with reasoning models:

1. DeepSeek-R1 via Vertex MAAS — chain-of-thought is emitted inline as
   ``<think>...</think>`` inside ``message.content`` (no separate field).
2. DeepSeek reasoner / OpenAI o-series / Anthropic extended thinking /
   Gemini thinking — litellm populates ``message.reasoning_content`` and
   leaves ``message.content`` clean.
3. Standard (non-reasoning) responses — must continue to round-trip unchanged
   (regression guard).

The ``<think>`` stripping and reasoning-capture lives on
:class:`buttermilk.utils.json_parser.ChatParser` — these tests exercise both
the parser-level surface and the integration through
:func:`buttermilk._core.llms.litellm_to_autogen_result`.
"""

from unittest.mock import MagicMock

from buttermilk._core.llms import litellm_to_autogen_result
from buttermilk.utils.json_parser import ChatParser


def _make_response(content, reasoning_content=None):
    """Build a minimal litellm-shaped response mock."""
    response = MagicMock()
    choice = MagicMock()
    # explicit defaults so MagicMock auto-attrs don't accidentally match
    choice.message.content = content
    choice.message.tool_calls = None
    if reasoning_content is None:
        # Force getattr() to return None rather than a MagicMock
        del choice.message.reasoning_content
    else:
        choice.message.reasoning_content = reasoning_content
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.cached = False
    response.model = "test-model"

    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 10
    return response, usage


def test_extract_reasoning_simple():
    """The method removes a single <think>...</think> block and records it."""
    parser = ChatParser()
    raw = '<think>\nAlright, let me think...\n</think>\n{"prediction": true}'
    cleaned = parser.extract_reasoning(raw)
    assert cleaned == '{"prediction": true}'
    assert parser.thought == "Alright, let me think..."


def test_extract_reasoning_no_tag_passthrough():
    """Plain text without <think> tags is returned unchanged and thought stays None."""
    parser = ChatParser()
    raw = '{"prediction": true}'
    cleaned = parser.extract_reasoning(raw)
    assert cleaned == '{"prediction": true}'
    assert parser.thought is None


def test_extract_reasoning_only_reasoning_falls_back():
    """If stripping leaves nothing, return the original (don't blank content).

    The reasoning is still captured so the caller can route it onto
    ``result.thought`` while downstream JSON parsing emits a diagnostic rather
    than receiving an empty string.
    """
    parser = ChatParser()
    raw = "<think>no answer was ever produced</think>"
    cleaned = parser.extract_reasoning(raw)
    assert cleaned == raw
    assert parser.thought == "no answer was ever produced"


def test_extract_reasoning_structured_takes_precedence():
    """When ``structured_reasoning`` is supplied, it wins over inline tags."""
    parser = ChatParser()
    raw = '<think>inline</think>\n{"x": 1}'
    cleaned = parser.extract_reasoning(raw, structured_reasoning="from-litellm")
    assert cleaned == '{"x": 1}'
    assert parser.thought == "from-litellm"


def test_deepseek_r1_vertex_maas_inline_think_block_is_stripped():
    """DeepSeek-R1 via Vertex MAAS: <think>...</think> inline in content.

    This is the exact failure mode reported by the user: the raw response
    starts with ``<think>\\nAlright, let's tackle this step by step...`` and
    the JSON parser fails because the reasoning block is concatenated with
    the answer.
    """
    reasoning_body = "Alright, let's tackle this step by step. The user has shared a news excerpt and wants me to analyze it..."
    raw_content = f'<think>\n{reasoning_body}\n</think>\n{{"prediction": true, "confidence": "high"}}'
    response, usage = _make_response(raw_content)

    result = litellm_to_autogen_result(response, usage, "deepseek-r1-maas")

    assert result.content == '{"prediction": true, "confidence": "high"}'
    # Reasoning surfaces on the proper CreateResult field, not in metadata.
    assert result.thought == reasoning_body
    assert "reasoning_content" not in result.metadata


def test_reasoning_content_field_is_preserved_in_thought():
    """DeepSeek reasoner / o-series / Anthropic thinking: reasoning_content
    is populated by litellm separately. Content is already clean JSON.
    """
    clean_content = '{"answer": "Paris"}'
    reasoning = "The user asks for the capital of France. That is Paris."
    response, usage = _make_response(clean_content, reasoning_content=reasoning)

    result = litellm_to_autogen_result(response, usage, "deepseek-reasoner")

    # Content untouched
    assert result.content == clean_content
    # Reasoning lands on result.thought (the CreateResult field)
    assert result.thought == reasoning
    assert "reasoning_content" not in result.metadata


def test_structured_reasoning_wins_over_inline_think_block():
    """When both signals are present, structured ``reasoning_content`` wins.

    In practice they almost never co-occur — a provider does one or the other —
    but the merge precedence is well-defined so that we never silently drop
    the cleaner, provider-curated value.
    """
    inline = "inline chain of thought"
    structured = "structured provider reasoning"
    raw_content = f'<think>{inline}</think>\n{{"prediction": false}}'
    response, usage = _make_response(raw_content, reasoning_content=structured)

    result = litellm_to_autogen_result(response, usage, "hybrid-model")

    # Inline block still stripped from content
    assert result.content == '{"prediction": false}'
    # Structured value wins
    assert result.thought == structured


def test_regression_plain_json_content_unchanged():
    """Standard (non-reasoning) responses must continue to pass through."""
    raw_content = '{"prediction": false, "labels": ["A", "B"]}'
    response, usage = _make_response(raw_content)

    result = litellm_to_autogen_result(response, usage, "gpt-4o")

    assert result.content == raw_content
    assert result.thought is None
    assert "reasoning_content" not in result.metadata
