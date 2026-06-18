# Redesign: PR #435 — cross-provider prompt caching that captures the *large reused* content

**Status:** design for review (supersedes the single-system-breakpoint implementation currently on PR #435)
**Scope:** `buttermilk/_core/llms.py`, `buttermilk/templates/prompt/score.jinja2`, tests. Wrapper-simplification (point 4) explicitly scoped *out* — see §6.
**Audience:** reviewers of PR #435 (closes `buttermilk-c6ffcb5c`, parent `buttermilk-e6c54701`).
**Code anchor:** the helper this redesign supersedes (`_add_anthropic_cache_control`) and the caching call-path live **only on PR #435 branch `polecat/buttermilk-c6ffcb5c` @ `ee50c343`** — they are *not* on `dev`. Review against that branch; grepping `dev` will not find them. All other line references (`to_litellm_messages`, `_create_client`, templates) are on `dev`.

This redesign responds to Nic's latest comment (the four numbered assumptions) and folds in the prior measured redesign. The headline: **Nic's intent — "one elegant solution that works across providers" — is achievable, but the mechanism Nic proposed (split each template variable into its own message) is the one structure our own live tests proved is *worst*: it breaks Gemini's cache entirely.** The elegant cross-provider solution is the opposite of splitting: *one* stable leading system block, with a `cache_control` stamp on it.

---

## 1. The four assumptions, checked against our own evidence

Nic's comment made four assumptions and flagged each as "assume / almost certainly / must be" — i.e. to be verified, not banked. They have been verified against (a) the live cache-variant matrix we ran on PR #432, (b) current provider docs (PKB `mem-712368f4`), and (c) the litellm 1.83 source.

### Assumption 1 — "each turn is individually and incrementally cached, so split each template variable into its own message"

**Disproven — and the proposed structure is actively harmful.** This is the exact hypothesis we already tested live (PR #432, `tests/test_prompt_cache_variants.py`, >5000-token prefix clearing every floor, three variants):

| Variant | Structure | gpt-5-nano | gemini-3.1-flash-lite | claude-haiku-4.5 |
|---|---|---|---|---|
| A — stuffed | reused + variable in one UserMessage | 5632 cached | **0** | 0 |
| **B — system block** | reused as one leading SystemMessage, variable last | 5632 cached | **4071 cached (HIT)** | 0 |
| C — same-role split | reused split across consecutive user messages | 5632 cached | **0** | 0 |

Caching is a **token-level prefix match**, not per-turn (verified against Anthropic, Gemini, Vertex, OpenAI docs — `mem-712368f4`). Splitting one variable per message does **not** make each message independently cacheable. Worse: variant C (the literal "split into messages" proposal) **zeroes Gemini's cache** — Gemini only matched the prefix when the reused content was a *single* leading SystemMessage (variant B). On OpenAI it was a no-op (A=B=C). Claude cached nothing in any variant because Claude caches nothing without an explicit `cache_control` breakpoint (that gap is exactly what PR #435 fixes).

**Conclusion:** message-splitting is not the lever and must not be adopted. The lever is *variant B*: the reused block as a single leading system block.

### Assumption 2 — "then it's easy to add cache_control at those boundaries"

**Half right.** `cache_control` is required for Anthropic (the only way any Claude model caches), and litellm translates it to each provider's native format. But it is *not* placed at per-message boundaries — it is a **prefix breakpoint** (Anthropic allows max 4), and adding the field on Gemini/OpenAI is simply ignored by those providers (they cache implicitly from prefix position). So the single useful breakpoint is on the *stable system block*, not on per-variable message boundaries.

### Assumption 3 — "this is probably already fixed / better in litellm; adopt best practice"

**Partly true, and there is a concrete win here.** litellm does **not** auto-cache for you and there is no single cross-provider switch. But litellm 1.83 ships `AnthropicCacheControlHook` + a declarative `cache_control_injection_points` param: you pass `cache_control_injection_points=[{"location": "message", "role": "system"}]` and litellm injects the ephemeral breakpoint onto the system message itself (`litellm/integrations/anthropic_cache_control_hook.py`). **This is the best-practice replacement for our hand-rolled `_add_anthropic_cache_control()` helper** — it deletes ~40 lines of string/list-content branching and is maintained upstream. It remains Anthropic/Bedrock-only translation (it does not unify Gemini/OpenAI — those need no field), so it is a *code-quality* adoption, not the cross-provider unifier. Adopt it (§5).

### Assumption 4 — "now we're litellm-only, strip out the model-specific wrappers and simplify greatly"

**Real opportunity, but mostly auth plumbing, not caching, and risky — scope it out.** The `client_type` branches in `LLMs._create_client()` (llms.py:1612–1659) are doing **GCP auth wiring** (access-token injection, `vertex_project`/`vertex_location`, token-refresh callbacks, `extra_headers`), not model-behaviour wrapping. litellm 1.83 *does* route `vertex_ai/claude-*`, `vertex_ai/llama-*`, etc. natively, but it still needs the project/location/credentials buttermilk injects, so the branches can be **consolidated** (the GEMINI/LLAMA/DEEPSEEK/MISTRAL-Vertex arms are nearly identical) but not deleted. This touches every provider in the roster and carries real regression risk across auth paths. It is **orthogonal to caching** and belongs in its own task/PR with its own roster verification — not folded into the caching change. See §6 for the recommendation.

---

## 2. The elegant cross-provider design (what Nic actually wants)

There **is** a single mechanism that works across every provider — it is just structural rather than per-message:

> **Assemble the large reused content (instructions + criteria) as one byte-identical leading *system block*, put all per-record variable content in trailing turns, and stamp `cache_control: ephemeral` on the system block.**

Why this is the universal solution:
- **Gemini / Vertex-Gemini:** caches the leading system block implicitly (variant B was the *only* variant that hit). The `cache_control` field is ignored harmlessly.
- **OpenAI / Vertex-OpenAI:** caches the common prefix implicitly; field ignored.
- **Anthropic / Anthropic-Vertex:** the `cache_control` stamp is the *only* thing that makes Claude cache; litellm translates it.
- **Llama / DeepSeek / Mistral-Vertex:** no caching mechanism exists at all — expected zero, excluded from assertions (`mem-712368f4`).

One structure, one stamp, every provider handled. No per-provider message juggling, no message-splitting.

**Canonical layout (judge step):**
```
[ system: instructions + CRITERIA ]   ◀ cache_control breakpoint   ← constant across all calls (~21k tok)
[ user:   RECORD / article ]                                        ← per-record variable (~840 tok), last
```
`judge.jinja2` *already has this shape* — instructions + `{{ render_or_include(criteria) }}` in the `# System:` block, `{{record}}` in a separate `# Placeholder:` turn. The breakpoint on the system block lands after the ~21k criteria. **Judge needs no template change** — PR #435's mechanism already works for it once the breakpoint is on the system block.

---

## 3. The one genuine template fix: `score.jinja2`

`score.jinja2` is the case the original objection was pointing at. Its `# System:` block **interleaves constant and variable content**:
```
# System:
  reviewer role
  {{ instructions }}   ← constant (carries criteria)
  {{ source }}         ← per-record article   ✗ variable, inside the cached prefix
  {{ expected }}       ← per-record key points ✗ variable, inside the cached prefix
# User:
  {{ answers }}        ← per-call
```
A breakpoint on this system block caches a prefix that **changes every record** (`source`/`expected`), so the reviewer step caches poorly or never. Fix — make the constant `instructions` a clean leading prefix and push the variable parts after the breakpoint:
```
# System:
  reviewer role + {{ instructions }}    ◀ cache_control breakpoint   ← constant
# User:
  {{ source }}                                                       ← per-record (own segment; 2nd breakpoint deferred — §4)
# User:
  {{ expected }}
  {{ answers }}                                                      ← variable, last
```
This is the only template that needs restructuring. (Gemini requires the trailing turn to be `user` and tolerates consecutive same-role user turns — confirmed live in PR #434's conformance test.)

---

## 4. Breakpoint budget

Anthropic gives 4 breakpoints. **This PR spends exactly one.**
- **Breakpoint 1 — end of the instructions/criteria system block** (for score: end of the `instructions` block). This is the entire win (~21k tok reused across all repeats × models × records sharing a criteria set) and it pays off *regardless of run-order* — the criteria cache is long-lived and reused across everything sharing a criteria set. Mandatory; verifiable in isolation; lands in this PR.
- Leave 3 spare. Do not spend them now.

**Breakpoint 2 (end of the record turn) is deliberately deferred out of this PR.** Its payoff is three conditionals deep and depends on a change in *another* repo: it only helps if an article's repeats fall within the cache TTL, which only holds if TJA adopts record-contiguous expansion order *or* a 1h TTL (Anthropic's default 5-min TTL will not hold a 120-way fan — §6). Buttermilk-core cannot verify that precondition, so breakpoint 2 is filed as a follow-up *gated on a measured TJA run-order change*, not banked here.

---

## 5. Code changes

1. **Replace the hand-rolled helper with litellm's declarative injection (point 3).** Delete `_add_anthropic_cache_control()` (the ~40-line string/list-content helper on the PR branch); instead, when `client_type in {anthropic, anthropic_vertex}`, pass `cache_control_injection_points=[{"location": "message", "role": "system"}]` to `litellm.acompletion` (the `AnthropicCacheControlHook` reads this param and stamps the breakpoint). Note: the hook injects into the message regardless of provider — the **Anthropic-only scope comes from buttermilk's own `client_type` gating (§5.2), not from the hook**; on Gemini/Vertex the field is ignored harmlessly (`mem-712368f4`). This is upstream-maintained and provider-translated. **Pre-merge hard check:** confirm `cache_control_injection_points` exists with this signature in the pinned litellm (verified present in the installed 1.83.0; see §9 on the pin).
2. **Keep** the `client_type` field on `LiteLLMWrapper` and the gating to Anthropic paths — these are correct and minimal.
3. **`score.jinja2` restructure** per §3.
4. **Floor guard:** when the would-be-cached prefix is below the model floor (4096 tok Opus/Haiku-4.5; 2048 Sonnet-4.6/Fable-5), `log.warning` rather than silently stamping an ineffective breakpoint. Size is knowable at assembly time. `trans_simplified` (831 tok) and other small criteria sets will trip this — see the reachability estimate in §8.
5. **Prefix-invalidator audit/test:** assert no per-call token (`run_number` from `pipeline_judge.yaml`, timestamps, UUIDs, agent IDs, unsorted JSON) is rendered *ahead* of breakpoint 1.
6. **No `messages.py` widening in this PR — but the gap is deferred, not resolved.** For judge and score, the reused content *can* be made a clean leading prefix (judge already is; score is reordered per §3), so ordering + the system-block structure suffice and no `messages.py` change is needed **here**. This is *not* a general claim that multi-part content is unnecessary: where a template's semantics fix content order so the reused block *cannot* be the leading prefix (Nic's 2026-06-18 note: "the record is sent LAST and may be large"), the only lever is multi-part single-turn content — and that is blocked by a real architectural gap (see §6a). Scoped out of this PR; explicitly *not* solved by it.

---

## 6. Point 4 (wrapper simplification) — recommended scope: separate task

The simplification Nic wants is real but is **auth-path consolidation, not caching**, and is the highest-regression-risk change in the area. Recommendation:
- **Do not** fold it into the caching PR. Caching must land verifiable on its own.
- File a separate task: "consolidate `_create_client` Vertex auth branches; confirm litellm 1.83 native routing for each `client_type` against the live roster." It needs a full-roster smoke test (every `client_type` constructs and calls), which the caching PR does not require.
- Likely outcome: the GEMINI/LLAMA/DEEPSEEK/MISTRAL-Vertex arms collapse into one "vertex-native" arm sharing the token-refresh + project/location wiring; ANTHROPIC_VERTEX and VERTEX_OPENAI keep their distinct auth. That is a meaningful tidy, not the wholesale deletion the phrasing implies.

### 6a. The architectural debt this PR routes around (named, not hidden)

This redesign is a **tactical fix at the cheapest, lowest-risk points** (a breakpoint param in `create()`; a template reorder). It deliberately does **not** touch the real root cause, which is the **content-typing limitation in the prompt-assembly layer**:
- `UserMessage.content` is typed `Union[str, list[str]]` (`buttermilk/_core/messages.py:56`) — it cannot carry content-part *objects*.
- `to_litellm_messages()` (`buttermilk/_core/llms.py:582`) emits one dict per message and cannot express multi-part content.

The cache breakpoint and the template shape are both downstream of these. For templates that can be reordered (judge, score) the tactical fix suffices. For any future template whose semantics fix content order, the proper fix is widening the content typing so reused content can be isolated as a stable segment regardless of position. **That is real architectural debt, owned by the prompt-assembly layer, and it is deferred — not closed — by this PR.** Capture it as a standalone task so the "not needed here" framing doesn't bury it.

---

## 7. What to keep from PR #435 vs. change

**Keep:** `client_type` field + Anthropic-only gating; the test scaffold (`TestAnthropicCacheControlInCreate` — retarget assertions to the litellm-injection path); the Gemini/OpenAI no-op tests; the deferral of live ADC verification to `buttermilk-ed61e1bc`.

**Change:** replace `_add_anthropic_cache_control()` with litellm `cache_control_injection_points` (§5.1); restructure `score.jinja2` (§3); add the floor guard (§5.4) and prefix-invalidator test (§5.5).

**Drop entirely:** any move toward one-variable-per-message splitting (§1, assumption 1) — it is disproven and breaks Gemini.

---

## 8. Revised acceptance criteria

1. Reused block (instructions + criteria) delivered as a **single leading system block**, byte-identical across calls, variable per-record content in trailing user turns — for **both** judge and score. (judge already conforms; score restructured.)
2. Anthropic path sets **breakpoint 1** on the system block via litellm's `cache_control_injection_points` (not a hand-rolled helper). (Breakpoint 2 / record is out of scope — §4.)
3. Floor guard warns (does not silently no-op) when the Anthropic cached prefix is below the model floor (4096 / 2048).
4. Prefix-invalidator test: no per-call token (`run_number`, timestamps, UUIDs, agent IDs) appears ahead of breakpoint 1.
5. Gemini/OpenAI unchanged & unregressed (the system-block structure already caches them); Llama/DeepSeek/Mistral excluded from cache assertions.
6. **Cost-reachability estimate recorded** (this is the prize Nic is buying — size it, don't assume it). Tabulate roster model × criteria-set size × per-model floor → "cache is reachable for N of M models and K of the 8 criteria sets." The known shape: criteria sets `tja` (20,788), `glaad` (10,333), `lgbtq_stylebook` (10,630) clear every floor; `apc` (3,448), `hrc` (2,493), `trans_factored` (2,138) clear the 2048 floor (Sonnet-4.6/Fable-5/Gemini) but **miss** the 4096 floor (Opus/Haiku-4.5); `cte` (1,347), `trans_simplified` (831) clear nothing for Anthropic. Llama/DeepSeek/Mistral never cache. State the resulting expected input-token-cost reduction (order-of-magnitude is fine) so the PR proves it serves the cost goal — confirm the floor-straddling sets with a real tokenizer, not char/4.
7. Live verification (`buttermilk-ed61e1bc`, re-run *after* this rework): `cache_read_input_tokens > 0` on a Claude judge call at a cache-eligible size; `cached_tokens > 0` on a Gemini call.
8. Point-4 wrapper consolidation filed as a **separate** task (§6); the `messages.py` content-typing debt (§6a) filed as a **separate** task. Neither in this PR.

### Deferred to follow-up (explicitly NOT in this PR)
- **Breakpoint 2 (record-level cache)** — gated on a measured TJA run-order change (§4); file TJA-side recommendation (record-contiguous expansion and/or 1h TTL) there.
- **`messages.py` content-typing widening** (§6a) — the architectural root cause; needed only for future templates that can't be reordered.
- **Vertex auth-branch consolidation** (point 4, §6).

---

## 9. Flagged / not settled

- Token counts (criteria ~21k, record ~840) are char/4 estimates from the rendered trace, not provider tokenizers — directionally exact (criteria ≫ record by ~25×); the **floor-straddling criteria sets** (`apc`/`hrc`/`trans_factored` near the 4096 line) are load-bearing for the §8 reachability estimate and **must** be confirmed with a real tokenizer before relying on them.
- **litellm pin (hard pre-merge):** the design depends on `cache_control_injection_points` (litellm ≥1.83). The declared pin in `pyproject.toml` is `litellm>=1.0.0`; the installed/locked version is 1.83.0 (where the param is verified present). **Raise the floor pin to `>=1.83`** so the mechanism isn't silently lost on a downgrade.
- litellm's `cache_control_injection_points` translation for `anthropic_vertex` specifically (vs. plain `anthropic`) should be smoke-tested — the hook runs for Anthropic; the Vertex-Anthropic route should be verified to emit the breakpoint to the wire (same transform factory, but confirm).
