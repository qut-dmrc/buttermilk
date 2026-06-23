"""Per-provider/model prompt-STYLE conformance probe.

Empirically answers (task ``buttermilk-410e31f7``): per provider/model, does the
Buttermilk call path accept

  (a) multiple consecutive SAME-speaker (user/user) messages, and/or
  (b) multiple PARTS within a single message/turn?

This gates the prompt-caching design (``buttermilk-e6c54701``): where prompt
order can't be changed, splitting into separate messages is the only way to
isolate a cacheable prefix — but only if the provider accepts the structure.

No mocks — REAL API calls through the real ``LiteLLMWrapper`` (``real_llm``
fixture, parametrized over ``CHEAP_CHAT_MODELS`` from conftest). One model
failing must NOT abort the suite or the other probes: each probe records
``ok`` or the exact provider error string and asserts nothing that would halt
collection of the matrix. A final summary is printed at session teardown.

Run:  uv run pytest tests/test_prompt_style.py -m integration -s -v -n 0

NOTE ON xdist: this repo runs pytest with ``-n auto`` by default. The summary
matrix below is accumulated in module-level state, which does NOT survive the
per-worker process boundary — so to see the aggregated matrix run with ``-n 0``
(single process). Under the default ``-n auto`` the per-probe lines still print
(with ``-s``), they are just spread across workers.
"""

from __future__ import annotations

import pytest

from buttermilk._core.messages import UserMessage

pytestmark = [pytest.mark.integration]

# ---------------------------------------------------------------------------
# Result accumulator. Keyed by model name -> probe -> "ok" | "<error string>".
# Printed as a matrix at session teardown so the whole roster is visible even
# when individual models error.
# ---------------------------------------------------------------------------
_RESULTS: dict[str, dict[str, str]] = {}

# Probe identifiers
P_CONTROL = "control(single-user)"
P_CONSECUTIVE = "consecutive-same-role"
P_MULTIPART = "multi-part-within-turn"


def _model_name(real_llm) -> str:
    """Best-effort human-readable model id for the matrix."""
    for attr in ("litellm_model_name", "model"):
        val = getattr(real_llm, attr, None)
        if val:
            return str(val)
    return repr(real_llm)


async def _probe(real_llm, messages) -> str:
    """Run one probe. Return 'ok' on success or the exact error string.

    Never raises — a provider rejection is the DATA we want, not a test
    failure, so we capture it and keep going.
    """
    try:
        result = await real_llm.create(messages=messages)
    except Exception as e:  # noqa: BLE001 — capturing provider errors is the point
        return f"{type(e).__name__}: {e}"

    # ProcessingError / content issues may be surfaced on the result object
    # rather than raised. Treat a populated error_message as a failure too.
    err = getattr(result, "error_message", None)
    if err:
        return f"result.error_message: {err}"
    if result is None or getattr(result, "content", None) in (None, ""):
        return "empty-content (no exception, but no usable response)"
    return "ok"


@pytest.mark.anyio
async def test_prompt_style_control(real_llm):
    """(c) Baseline: one plain user message. Establishes the model works at all."""
    name = _model_name(real_llm)
    _RESULTS.setdefault(name, {})
    messages = [
        UserMessage(content="What is 2+2? Answer with just the number.", source="test"),
    ]
    outcome = await _probe(real_llm, messages)
    _RESULTS[name][P_CONTROL] = outcome
    print(f"\n[{name}] {P_CONTROL}: {outcome}")


@pytest.mark.anyio
async def test_prompt_style_consecutive_same_role(real_llm):
    """(a) Two consecutive same-role (user) messages, no assistant turn between.

    Strict-alternation providers (e.g. Gemini/Vertex) are expected to reject or
    silently merge; Anthropic merges; OpenAI is flexible. We record the provider
    behaviour rather than asserting a particular outcome.
    """
    name = _model_name(real_llm)
    _RESULTS.setdefault(name, {})
    messages = [
        UserMessage(content="Here is some context: the capital of France is Paris.", source="test"),
        UserMessage(content="What is the capital of France? One word.", source="test"),
    ]
    outcome = await _probe(real_llm, messages)
    _RESULTS[name][P_CONSECUTIVE] = outcome
    print(f"\n[{name}] {P_CONSECUTIVE}: {outcome}")


@pytest.mark.anyio
async def test_prompt_style_multipart(real_llm):
    """(b) A single user message carrying multiple content PARTS in one turn.

    NOTE ON THE CALL PATH (gap analysis): buttermilk's ``UserMessage.content`` is
    typed ``str | list[str]`` — it CANNOT express OpenAI-style content-part dicts
    (``[{"type":"text",...}]``); Pydantic rejects them. ``to_litellm_messages``
    passes ``content`` straight through. So the ONLY multi-part-within-turn form
    the real call path can express is a LIST OF STRINGS. That is exactly what we
    probe here — the question is whether each provider accepts a list-valued
    ``content`` on a single user turn.
    """
    name = _model_name(real_llm)
    _RESULTS.setdefault(name, {})
    messages = [
        UserMessage(
            content=[
                "Here is some context: the capital of France is Paris.",
                "What is the capital of France? One word.",
            ],
            source="test",
        ),
    ]
    outcome = await _probe(real_llm, messages)
    _RESULTS[name][P_MULTIPART] = outcome
    print(f"\n[{name}] {P_MULTIPART}: {outcome}")


@pytest.fixture(scope="session", autouse=True)
def _print_matrix():
    """Emit the capability matrix at session teardown (visible with -s)."""
    yield
    if not _RESULTS:
        return
    probes = [P_CONTROL, P_CONSECUTIVE, P_MULTIPART]
    print("\n\n" + "=" * 100)
    print("PROMPT-STYLE CONFORMANCE MATRIX  (model x probe -> ok / provider error)")
    print("=" * 100)
    for model in sorted(_RESULTS):
        print(f"\n### {model}")
        for probe in probes:
            val = _RESULTS[model].get(probe, "(not run)")
            status = "OK " if val == "ok" else "ERR"
            print(f"  [{status}] {probe}: {val}")
    print("\n" + "=" * 100)
