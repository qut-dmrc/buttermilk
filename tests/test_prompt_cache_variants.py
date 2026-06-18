"""Live caching-verification test: stuffed vs. separate-component prompts.

Parameterised across buttermilk's roster via the canonical ``real_llm`` fixture
(``@pytest.fixture(params=CHEAP_CHAT_MODELS)`` in ``tests/conftest.py``), which
yields one live client per cheap model. Pytest expands one parametrised case
per model automatically — the test does NOT build any model list of its own;
``CHEAP_CHAT_MODELS`` is the single source of truth. This test empirically
verifies prompt-cache behaviour by comparing two prompt-assembly styles.

Variant A — "stuffed"
    One single UserMessage containing the large reused prefix immediately
    followed by the small variable query.  Simulates monolithic prompt style.

Variant B — "separate components"
    The reused block delivered as a discrete SystemMessage; the variable content
    as a trailing UserMessage.  Simulates the structured-prefix style that
    providers can implicitly cache.

For each (model × variant) pair the test calls the client TWICE with identical
messages and records:
  - cached_tokens on each call (via extract_cached_tokens from pricing.py)
  - total_cost on each call (from pricing metadata returned by LiteLLMWrapper)
  - whether call-2 shows a cache hit (cached_tokens > 0)

The test REPORTS results in a matrix rather than asserting all hits; some
providers do not cache under certain assembly styles, and that is the finding.
Errors on Variant B (esp. Gemini/Vertex alternation constraints) are captured
and reported — not raised — so the full matrix is always collected.
"""

from __future__ import annotations

import pytest

from buttermilk._core.messages import SystemMessage, UserMessage
from buttermilk.utils.pricing import extract_cached_tokens

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# ─────────────────────────────────────────────────────────────────────────────
# Filler block: must exceed per-provider implicit-cache token minimums.
#   Gemini / Vertex:  2048 tokens
#   OpenAI:           1024 tokens
#   Anthropic:        1024–4096 tokens
# Estimated at ~4 000–5 000 tokens of English text (~18 000 characters).
# ─────────────────────────────────────────────────────────────────────────────
_CRITERIA_BLOCK = """\
CONTENT MODERATION CRITERIA — RESEARCH INSTRUMENT v1.0
=======================================================

This document defines the criteria used by human raters and AI models to
assess potentially harmful or sensitive social-media content. All decisions
must be grounded in the explicit text or imagery present; speculation about
author intent is not permitted.

SECTION 1 — HATE SPEECH AND DISCRIMINATION
-------------------------------------------
1.1  Racial and ethnic targeting
     Content that derogates, threatens, or incites violence against individuals
     or groups on the basis of race, ethnicity, national origin, or caste.
     Indicators: slurs, dehumanising metaphors (vermin, disease, infestation),
     calls for exclusion or violence, neo-Nazi or white-nationalist iconography,
     Holocaust denial, or comparisons equating ethnic groups with animals.
     The assessment must consider both explicit and coded language.  Coded
     language ("dog whistles") that a reasonable observer from the targeted
     community would recognise as derogatory should be treated equivalently
     to explicit slurs when the surrounding context makes the intent clear.
     Historical or educational quotation of slurs in clearly analytical contexts
     does not itself constitute a criterion violation, but must be flagged for
     contextual review.

1.2  Religious hatred
     Attacks on adherents of a religion that go beyond critique of doctrine.
     Distinct from lawful criticism of religious institutions or ideas.
     Indicators: calls to harm worshippers, desecration of sacred symbols in a
     derogatory context, conspiracy theories targeting a faith community.
     The distinction between theological disagreement (permitted) and incitement
     against believers (prohibited) turns on whether the content targets the
     ideas of the faith or the people who hold them.

1.3  Gender and sexual orientation discrimination
     Misogynistic, transphobic, or homophobic content.  Includes deadnaming,
     non-consensual outing, denial of gender identity, and targeted harassment
     of LGBTQ+ individuals.  Distinguished from good-faith theological debate
     about doctrine, though such debate may still be flagged for contextual
     review at severity 1 where it risks normalising discrimination.
     Content that sexualises or objectifies individuals on the basis of gender
     should be rated under criterion 1.3 in addition to any applicable
     criterion from Section 2.

1.4  Disability discrimination
     Mocking or dehumanising people with physical, cognitive, or psychiatric
     disabilities.  Includes ableist slurs and the promotion of "inspiration
     porn" that objectifies disabled individuals.  Medical misinformation that
     disproportionately harms people with disabilities (e.g., anti-vaccine
     content targeting immunocompromised individuals) should be co-coded under
     criterion 1.4 and the relevant criterion from Section 3.

1.5  Intersectionality
     Content that compounds discrimination across multiple dimensions (e.g.,
     racist misogyny, anti-Semitic homophobia).  Each dimension axis should be
     coded separately; the presence of one axis does not preclude the other.
     The overall severity rating reflects the most severe single axis, but the
     annotation record must enumerate all axes present.

1.6  Socioeconomic discrimination and classism
     Content that degrades, excludes, or incites hostility toward individuals
     on the basis of their economic status, occupation, or social class.
     Distinct from legitimate critique of economic systems or policy positions.

SECTION 2 — VIOLENCE AND GRAPHIC CONTENT
-----------------------------------------
2.1  Incitement to violence
     Direct or indirect calls for violence against named or readily identifiable
     individuals or groups.  Includes coded language (e.g., "somebody should do
     something") where context makes the threatening intent evident.  For public
     figures, the threshold is the same as for private individuals; public status
     does not diminish the risk of real-world harm from credible threats.

2.2  Graphic violence
     Depictions of severe injury, death, or torture presented without clear
     journalistic, educational, or artistic purpose.  Consider the platform
     context and whether a content warning is provided.  Newsworthy imagery of
     conflict or disaster may be rated at severity 1 rather than 2 where the
     public-interest value is evident and proportionate to the graphic content.

2.3  Self-harm and suicide
     Content that promotes, glorifies, or provides detailed methods for
     self-harm or suicide.  Safe-messaging guidelines require that such content
     include crisis-resource information.  Content that discusses suicidality
     in a harm-reduction or support-seeking frame should be distinguished from
     promotional content; the former is generally severity 0-1, the latter 3.

2.4  Child sexual abuse material (CSAM)
     Any sexualised depiction of a minor, real or simulated.  Zero-tolerance;
     must be escalated immediately regardless of artistic framing.  AI-generated
     imagery depicting minors in sexual situations is treated identically to
     photographic CSAM under these criteria.

2.5  Animal cruelty
     Gratuitous depictions of cruelty to animals presented for entertainment or
     shock value.  Distinguishable from hunting, farming, or veterinary contexts
     where harm to animals is incidental to a legitimate activity.

SECTION 3 — MISINFORMATION AND MANIPULATION
--------------------------------------------
3.1  Health misinformation
     False or unsupported claims about medical conditions, treatments, vaccines,
     or public-health guidance that could cause harm if acted upon.  Distinguish
     from legitimate scientific uncertainty or dissenting expert opinion: content
     that accurately characterises ongoing scientific debate is not misinformation
     even if the majority view is not represented.  Claims that contradict
     scientific consensus without evidential basis are misinformation.

3.2  Election and civic misinformation
     False claims about voting procedures, candidate eligibility, electoral
     outcomes, or the integrity of election systems, where the false claim is
     likely to suppress participation or delegitimise outcomes.  Satire is
     excluded where it is clearly labelled and unlikely to be mistaken for fact.

3.3  Synthetic-media disclosure
     AI-generated or manipulated images/video presented as authentic.  Content
     should be flagged when (a) there is no disclosure label and (b) the subject
     matter is likely to mislead a reasonable observer.  Clearly fantastical or
     stylised AI imagery (e.g., digital art) is not covered by this criterion
     even in the absence of a disclosure label.

3.4  Coordinated inauthentic behaviour
     Indicators of bot networks, astroturfing, or sock-puppet campaigns:
     identical or near-identical posts from multiple accounts in rapid
     succession, implausible posting frequencies, or disclosed purchase of
     engagement metrics.  Organic virality of genuine content is excluded.

3.5  Financial fraud and scams
     Content designed to deceive individuals into surrendering money or
     financial credentials.  Includes advance-fee fraud, phishing, pump-and-dump
     schemes, and fraudulent investment advice.  Educational content about fraud
     typologies is excluded.

SECTION 4 — PRIVACY AND PERSONAL DATA
--------------------------------------
4.1  Doxxing
     Non-consensual publication of private identifying information: home
     address, phone number, workplace, family members' identities, or other
     data that could enable stalking or harassment.  Publicly available
     information voluntarily disclosed by the subject is not doxxing, but
     aggregation of individually public data points that creates a harmful
     profile may still meet this criterion.

4.2  Revenge pornography / non-consensual intimate imagery (NCII)
     Non-consensual publication of intimate imagery, whether photographic or
     AI-generated.  Includes both original photographs/video and AI-generated
     depictions of identifiable real individuals in sexual situations.

4.3  Outing
     Disclosure of a person's sexual orientation, gender identity, immigration
     status, religion, mental-health history, or other private matters without
     consent, where disclosure could cause harm.  Applies regardless of whether
     the information is technically discoverable elsewhere.

4.4  Surveillance and tracking
     Content that facilitates covert monitoring of individuals without their
     knowledge or consent, including stalkerware promotion, location tracking
     of intimate partners, and collection of biometric data without disclosure.

SECTION 5 — SEVERITY SCALE
---------------------------
Each item is rated on a five-point severity scale:
  0 — No concern.  Content is benign or clearly within platform norms.
  1 — Borderline.  Context-dependent; unlikely to cause direct harm but
      warrants human review before amplification.
  2 — Moderate.  Clearly problematic; requires action such as a content
      warning, reduced distribution, or referral to platform trust-and-safety.
  3 — Severe.  Causes or is likely to cause significant harm to identifiable
      individuals or groups.  Requires removal and potential escalation.
  4 — Critical.  Immediate escalation required (e.g., CSAM, credible imminent
      threat to life, coordinated mass-harm campaign).

When an item violates multiple criteria at different severity levels, the
overall rating is the maximum severity across all applicable criteria.
When in doubt between two adjacent severity levels, choose the lower.

SECTION 6 — ANNOTATION GUIDELINES
-----------------------------------
6.1  Rate the content as presented, not as you imagine the creator intended.
6.2  When in doubt between two adjacent severity levels, choose the lower.
6.3  Record the primary criterion violated; note secondary criteria if present.
6.4  If the item lacks sufficient context to assess, mark it as "insufficient
     context" rather than guessing.
6.5  Demographic characteristics of the rater must not influence the rating.
6.6  Apply these criteria consistently regardless of the political or
     ideological orientation of the content.
6.7  Document your reasoning for every rating above 0.  Reasoning should be
     one to three sentences and reference specific textual or visual evidence.
6.8  When the content contains elements that appear satirical, apply the
     "reasonable observer" test: would a typical member of the targeted group
     recognise it as satire?  If not, rate on face value.
6.9  Platform context (e.g., adult content site vs. children's education
     platform) should inform severity but not criterion applicability.
6.10 Do not rate the same item more than once.  If you encounter an item you
     have previously rated, note the duplication and skip.

SECTION 7 — QUALITY ASSURANCE
-------------------------------
All items rated at severity 3 or 4 require secondary review by a senior
annotator before the rating is finalised.  Items rated at severity 2 by
two or more annotators independently are escalated for adjudication.
Inter-rater reliability is computed monthly; annotators whose agreement
falls below 0.70 (Cohen's κ) receive additional calibration training.

SECTION 8 — ILLUSTRATIVE EXAMPLES (non-exhaustive)
-----------------------------------------------------
Example A (Severity 0): "I strongly disagree with this politician's economic
policy."  Lawful political expression; no criterion violated.

Example B (Severity 1): A meme that implies, without explicit statement, that
members of a particular nationality are criminal-minded.  Borderline criterion
1.1; context-dependent.

Example C (Severity 2): An image macro combining an ethnic slur with a cartoon
stereotype.  Criterion 1.1 applies; moderate severity given no direct
incitement to violence.

Example D (Severity 3): A post listing the home address of a public official
alongside a statement that "someone should pay them a visit."  Criteria 2.1
and 4.1 apply; severe due to proximity of threatening language and doxxing.

Example E (Severity 4): An image depicting a minor in a sexual situation.
Criterion 2.4; critical escalation required regardless of artistic framing.

END OF CRITERIA DOCUMENT
"""

# Small variable suffix that changes the query while keeping the large prefix
# identical across both calls (enabling cache hits on the second invocation).
_VARIABLE_QUERY = (
    "Using the criteria above, assign a severity score (0-4) for the following "
    "post and cite the primary criterion that applies:\n\n"
    '    "I love everyone and wish the whole world peace and happiness."\n\n'
    "Reply with: Score: <n> | Criterion: <section> | Reason: <one sentence>."
)


# ─────────────────────────────────────────────────────────────────────────────
# Message constructors (called fresh before each (model × variant) run)
# ─────────────────────────────────────────────────────────────────────────────


def _variant_a_messages() -> list:
    """Variant A — stuffed: single UserMessage containing prefix + query."""
    return [
        UserMessage(
            content=_CRITERIA_BLOCK + "\n\n---\n\n" + _VARIABLE_QUERY,
            source="user",
        )
    ]


def _variant_b_messages() -> list:
    """Variant B — separate: SystemMessage for prefix, UserMessage for query."""
    return [
        SystemMessage(content=_CRITERIA_BLOCK),
        UserMessage(content=_VARIABLE_QUERY, source="user"),
    ]


_VARIANTS: dict[str, object] = {
    "A_stuffed": _variant_a_messages,
    "B_separate": _variant_b_messages,
}


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────


async def test_prompt_cache_matrix(real_llm, real_bm, session_runner) -> None:
    """Verify prompt-cache behaviour for one roster model × two variants.

    Parametrised by the canonical ``real_llm`` fixture (one live client per
    cheap model). For each variant, issues two identical calls and reports
    cached_tokens and cost delta. Never hard-fails on cache misses — the
    purpose is discovery, not assertion. Errors on Variant B (e.g. Gemini/Vertex
    message-structure constraints) are captured and reported per-model so the
    full matrix is always collected.
    """
    client = real_llm
    model_name: str = getattr(client, "model", None) or str(client)

    # row[variant_name] = result dict
    rows: dict[str, dict] = {}

    for variant_name, make_msgs in _VARIANTS.items():
        # Identical message list used for both calls.
        messages = make_msgs()  # type: ignore[operator]
        row: dict = {}
        try:
            # ── Call 1: warm the cache ────────────────────────────────
            r1 = await client.create(messages=messages, max_tokens=64)
            c1 = extract_cached_tokens(r1.usage)
            p1: dict = r1.metadata.get("pricing") or {}
            cost1: float = p1.get("total_cost") or 0.0

            # ── Call 2: expect cache hit on second request ────────────
            r2 = await client.create(messages=messages, max_tokens=64)
            c2 = extract_cached_tokens(r2.usage)
            p2: dict = r2.metadata.get("pricing") or {}
            cost2: float = p2.get("total_cost") or 0.0
            prompt2: int = getattr(r2.usage, "prompt_tokens", 0) or 0

            row = {
                "call1_cached": c1,
                "call2_cached": c2,
                "prompt_tokens": prompt2,
                "cost1": cost1,
                "cost2": cost2,
                "cost_drop": cost1 - cost2,
                "hit": c2 > 0,
            }
        except Exception as exc:
            row = {"error": str(exc)[:400]}

        rows[variant_name] = row

    _report(model_name, rows)

    # Structural assertion: a result row must exist for every variant.
    # This confirms the test ran (even if the model errored or missed cache).
    assert set(rows.keys()) == set(_VARIANTS.keys()), (
        f"Incomplete result for model '{model_name}': got {set(rows.keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helper
# ─────────────────────────────────────────────────────────────────────────────


def _report(model_name: str, rows: dict[str, dict]) -> None:
    sep = "=" * 122
    print(f"\n{sep}")
    print(f"PROMPT CACHE VERIFICATION — model: {model_name}")
    print("  Variant A = stuffed (single user message)   Variant B = separate (SystemMessage + UserMessage)")
    print(sep)
    print(f"{'Model':<46} {'Variant':<12} {'Cached1':>8} {'Cached2':>8} {'Cost1':>11} {'Cost2':>11} {'Drop':>11} {'Hit':>5}")
    print("-" * 122)
    for variant in ("A_stuffed", "B_separate"):
        row = rows.get(variant, {})
        if "error" in row:
            err_preview = row["error"][:64]
            print(f"{model_name:<46} {variant:<12}  ERROR: {err_preview}")
        else:
            hit_str = "YES" if row.get("hit") else "no"
            print(
                f"{model_name:<46} {variant:<12}"
                f" {row.get('call1_cached', 0):>8}"
                f" {row.get('call2_cached', 0):>8}"
                f" ${row.get('cost1', 0.0):>10.6f}"
                f" ${row.get('cost2', 0.0):>10.6f}"
                f" ${row.get('cost_drop', 0.0):>10.6f}"
                f" {hit_str:>5}"
            )
    print(sep)
