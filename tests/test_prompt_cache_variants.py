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

Variant C — "separate same-role messages"
    The reused block split across multiple consecutive same-role (user) messages,
    with the variable content as the final user message.  PR #434 proved all
    cheap providers ACCEPT consecutive same-role messages; this variant tests
    whether that split actually survives litellm to the wire as a distinct
    cacheable prefix (same benefit as A/B) or gets merged/wasted (less benefit).

The reused prefix (criteria document + extended appendix) is >5000 tokens — well
clear of every provider's implicit-cache floor (OpenAI 1024, Gemini 2048-4096,
Anthropic-Haiku 4096) — so a reported non-hit is a genuine non-hit rather than a
prefix that was simply too short to be cacheable.

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

from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import (
    LiteLLMWrapper,
    ModelInfo,
    litellm_to_model_output,
)
from buttermilk._core.messages import SystemMessage, UserMessage
from buttermilk.utils.pricing import extract_cached_tokens

# NOTE: module-level pytestmark removed in the #432+#436 union so that the
# reasoning-model UNIT tests (TestNullMessageDefensiveParse) are NOT marked
# integration. The live matrix + integration classes carry their own marks.

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

# ─────────────────────────────────────────────────────────────────────────────
# Appendix block — appended verbatim to _CRITERIA_BLOCK so the reused prefix
# comfortably clears the LARGEST provider implicit-cache floor.
#   OpenAI:            1024 tokens
#   Gemini 2.x/3.x:    2048 tokens (implicit), 4096 (some 3.x tiers)
#   Anthropic Haiku:   4096 tokens (min cacheable prefix)
# The prior ~2560-token prefix only cleared OpenAI's 1024 floor; Gemini-3.x and
# Anthropic-Haiku floors (4096) were never exercised, so their zeros were
# meaningless. This appendix lifts the combined prefix to ~6000+ tokens, well
# clear of every floor, so a non-hit is a genuine non-hit. The content is a
# fixed constant — identical across all calls — preserving prefix cacheability.
# ─────────────────────────────────────────────────────────────────────────────
_CRITERIA_APPENDIX = """\

APPENDIX A — EXTENDED ADJUDICATION CASEBOOK
=============================================

The following annotated cases supplement Section 8. Each entry states the raw
content (paraphrased), the criteria triggered, the assigned severity, and the
adjudication rationale. Annotators should treat these as calibration anchors,
not as an exhaustive enumeration of every possible fact pattern. Where a live
item resembles a casebook entry but differs in a material respect, annotators
must reason from the criteria in Sections 1–7 rather than mechanically copying
the casebook severity.

A.1  Coded ethnic dehumanisation in apparent satire
     A post frames an ethnic group using an extended pest-control metaphor,
     closing with an "it's just a joke" disclaimer. Applying the reasonable-
     observer test from 6.8, a typical member of the targeted community would
     recognise the dehumanising frame regardless of the disclaimer. Criterion
     1.1 applies; severity 2, escalating to 3 if combined with any call to
     action. The satire disclaimer does not lower severity because the metaphor
     itself performs the dehumanisation independent of comedic framing.

A.2  Doctrinal critique adjacent to incitement
     A long essay critiques the theology of a faith. The first nine paragraphs
     are lawful doctrinal disagreement (criterion 1.2 not met). The tenth
     paragraph pivots to urging followers be "removed from public life." The
     pivot triggers criterion 1.2 and, depending on specificity, 2.1. The
     overall item is rated at the maximum applicable severity (3), but the
     annotation record must note that the bulk of the text is permitted
     critique — this distinction matters for any downstream appeal.

A.3  Deadnaming in a news-reporting context
     A news article reports on a public figure and includes their former name
     in a single parenthetical for identification. Criterion 1.3 (deadnaming)
     is assessed against the public-interest carve-out: where the former name
     is necessary for unambiguous identification and is not used derogatorily,
     severity is 0–1. Repeated, gratuitous use of the former name shifts the
     assessment toward severity 2 and signals harassment intent.

A.4  Inspiration-porn framing of a disabled athlete
     A viral video frames a disabled athlete solely as an object of able-bodied
     motivation ("if they can do it, what's your excuse?"). Criterion 1.4
     applies at severity 1: the framing objectifies, but the content is not
     itself hateful. Annotators must resist the instinct to rate uplifting-
     seeming content at 0 merely because its surface affect is positive; the
     objectification is the harm.

A.5  Compounded racist misogyny
     A thread targets a named woman of colour with both racial slurs and
     gendered sexual degradation. Per criterion 1.5, both axes are coded
     separately; the annotation enumerates 1.1 and 1.3. The overall severity
     is the maximum single-axis severity (here 3), not a sum — but suppressing
     either axis in the record understates the harm profile and is an error.

A.6  Borderline self-harm support content
     A user describes their own past suicidality in an explicitly recovery-
     oriented frame and links a crisis resource. Criterion 2.3 distinguishes
     harm-reduction/support-seeking content (severity 0–1) from promotional or
     method-providing content (severity 3). The presence of the crisis resource
     and the retrospective framing place this at severity 0–1. Removing such
     content would itself cause harm by suppressing peer support.

A.7  Newsworthy graphic conflict imagery
     A photojournalist's image of a conflict casualty is posted with a content
     warning and contextual caption. Criterion 2.2's public-interest carve-out
     applies: severity 1 rather than 2, given evident newsworthiness and the
     proportionate use of the graphic element. The same image stripped of
     context and posted for shock value would be severity 2.

A.8  AI-generated synthetic political media
     A realistic AI-generated video depicts a candidate making statements they
     never made, with no disclosure label, days before an election. Criteria
     3.2 and 3.3 both apply. The absence of disclosure (3.3 condition a) and
     the likelihood of misleading a reasonable observer (3.3 condition b) are
     both satisfied; the electoral-suppression potential engages 3.2. Severity
     3, escalating toward 4 if coordinated distribution (3.4) is evident.

A.9  Aggregated public data constituting a harmful profile
     An account compiles a target's individually-public data points — employer,
     gym schedule, children's school — into a single dossier with hostile
     commentary. Although each datum is individually public, criterion 4.1's
     aggregation clause applies: the compilation creates a harmful profile
     enabling stalking. Severity 3. The "it was all public anyway" defence does
     not defeat the criterion.

A.10 Educational fraud-typology explainer
     A consumer-protection article explains how advance-fee scams operate so
     readers can recognise them. Criterion 3.5's educational carve-out applies;
     severity 0. Annotators must distinguish description-for-defence (permitted)
     from instruction-for-perpetration (prohibited): the former empowers
     potential victims, the latter equips offenders.

APPENDIX B — EDGE-CASE DECISION PROCEDURE
===========================================

When an item does not map cleanly onto a single criterion, apply the following
ordered procedure. Do not skip steps; record which step resolved the case.

B.1  Identify every criterion whose indicators are present, however weakly.
     Err toward over-inclusion at this enumeration stage; pruning happens later.

B.2  For each enumerated criterion, ask whether a carve-out (educational,
     newsworthy, artistic, harm-reduction, doctrinal-critique) plausibly
     applies. A carve-out lowers severity within a criterion; it does not
     remove the criterion from the record.

B.3  Apply the reasonable-observer test (6.8) to any criterion whose
     applicability turns on interpretation (satire, coded language, dog
     whistles). The relevant observer is a typical member of the targeted
     group, not the median platform user.

B.4  Resolve severity per criterion using Section 5, defaulting to the lower
     of two adjacent levels under genuine uncertainty (6.2).

B.5  Set the item's overall severity to the maximum across surviving criteria.
     Enumerate all surviving criteria in the record even though only the
     maximum drives the action.

B.6  If, after this procedure, the item cannot be assessed for lack of context,
     mark "insufficient context" (6.4) rather than guessing. Insufficient-
     context items are routed to human escalation, not auto-actioned.

B.7  Document the deciding step and the textual or visual evidence relied upon
     (6.7). A rating without recorded evidence is invalid and will be returned.

APPENDIX C — RATER CALIBRATION NOTES
======================================

C.1  The single most common error is criterion substitution: rating the
     content the annotator wishes were present rather than the content actually
     present (6.1). Re-read the raw item before finalising.

C.2  The second most common error is affect-driven leniency: rating
     positive-affect content (uplifting framing, humour, apparent good
     intentions) below its true severity. Surface affect is not a carve-out.

C.3  The third most common error is aggregation blindness: treating each
     individually-public or individually-mild element as benign while missing
     the compounded harm (see A.5, A.9). Assess the whole as well as the parts.

C.4  Annotators must apply criteria identically regardless of the political,
     ideological, or religious orientation of the content (6.6). Symmetry of
     application is a non-negotiable quality requirement; asymmetry is itself a
     reportable QA failure.

C.5  Inter-rater reliability below Cohen's κ = 0.70 triggers recalibration
     (Section 7). Annotators should periodically self-audit against this
     casebook to detect drift before the monthly κ computation surfaces it.

APPENDIX D — CRITERION-BY-CRITERION SEVERITY ANCHORS
======================================================

The following table fixes a default severity anchor for each criterion under
"typical" conditions, together with the contextual factors that move the anchor
up or down. Anchors are starting points, not ceilings or floors; the ordered
procedure in Appendix B always governs the final rating. The anchors exist to
reduce drift, not to replace judgement.

D.1  Criterion 1.1 (racial/ethnic targeting). Default anchor: 2. Moves to 3
     when paired with any incitement (2.1) or call for exclusion; moves to 1
     when the targeting is implicit/coded and a reasonable observer would
     require contextual knowledge to perceive it. Holocaust denial anchors at 3
     irrespective of tone, given its documented role in organised hate.

D.2  Criterion 1.2 (religious hatred). Default anchor: 2. Moves to 0–1 when the
     content is doctrinal critique that does not target believers; moves to 3
     when it urges harm to or exclusion of worshippers. The pivot test from A.2
     governs mixed essays: rate at the maximum applicable, record the split.

D.3  Criterion 1.3 (gender/orientation). Default anchor: 2. Deadnaming in a
     necessary-identification context anchors at 0–1 (see A.3); repeated
     gratuitous deadnaming or non-consensual outing anchors at 2–3. Targeted
     harassment campaigns anchor at 3.

D.4  Criterion 1.4 (disability). Default anchor: 2. Inspiration-porn framing
     anchors at 1 (see A.4); ableist slurs anchor at 2; medical misinformation
     disproportionately harming disabled people is co-coded with Section 3 and
     anchors at the higher of the two.

D.5  Criterion 1.5 (intersectionality). No independent anchor: severity is the
     maximum across the enumerated axes. The criterion's function is to force
     enumeration of every axis, not to add severity.

D.6  Criterion 1.6 (socioeconomic/class). Default anchor: 1. Moves to 2 when it
     incites hostility toward a class of persons rather than critiquing a
     system or policy. Legitimate policy critique anchors at 0.

D.7  Criterion 2.1 (incitement). Default anchor: 3. Moves to 4 when the threat
     is specific, credible, and imminent; moves to 2 when the call is diffuse
     and non-specific but still threatening. Public-figure status does not
     reduce the anchor (see body text of 2.1).

D.8  Criterion 2.2 (graphic violence). Default anchor: 2. Newsworthy, content-
     warned, proportionate imagery anchors at 1 (see A.7); shock-value reposts
     stripped of context anchor at 2; torture depicted for entertainment
     anchors at 3.

D.9  Criterion 2.3 (self-harm/suicide). Default anchor: 3 for promotional or
     method-providing content; 0–1 for harm-reduction or support-seeking
     content carrying crisis resources (see A.6). The frame, not the topic,
     drives the anchor.

D.10 Criterion 2.4 (CSAM). Fixed at 4. No carve-out, no contextual reduction,
     no artistic-framing exception. AI-generated and photographic material are
     treated identically. Immediate escalation; do not attempt independent
     adjudication.

D.11 Criterion 2.5 (animal cruelty). Default anchor: 2 for gratuitous shock-
     value cruelty; 0 for incidental harm in hunting, farming, or veterinary
     contexts. Staged cruelty for entertainment anchors at 3.

D.12 Criterion 3.1 (health misinformation). Default anchor: 2. Moves to 3 when
     the false claim is actionable and likely to cause physical harm (e.g.,
     fake cures for serious illness). Accurate characterisation of genuine
     scientific debate anchors at 0.

D.13 Criterion 3.2 (election/civic). Default anchor: 2. Moves to 3 when the
     false claim is likely to suppress participation or delegitimise outcomes;
     clearly-labelled satire anchors at 0.

D.14 Criterion 3.3 (synthetic-media disclosure). Default anchor: 1 for
     undisclosed synthetic media that could mislead; 0 for clearly fantastical
     or stylised AI imagery. Co-codes with 3.2 in electoral contexts (see A.8).

D.15 Criterion 3.4 (coordinated inauthentic behaviour). Default anchor: 2.
     Moves to 3 when CIB amplifies harmful content (hate, election
     misinformation). Organic virality is excluded and anchors at 0.

D.16 Criterion 3.5 (fraud/scams). Default anchor: 3 for active deception
     designed to extract money or credentials; 0 for educational explainers
     (see A.10).

D.17 Criterion 4.1 (doxxing). Default anchor: 3. The aggregation clause (see
     A.9) extends the criterion to harmful compilations of individually-public
     data. Single accidental disclosures with no hostile intent anchor at 1.

D.18 Criterion 4.2 (NCII). Default anchor: 3, escalating toward 4 where the
     subject is identifiable and distribution is wide. AI-generated NCII of real
     people is treated identically to photographic NCII.

D.19 Criterion 4.3 (outing). Default anchor: 2–3 depending on the harm exposure
     created by disclosure. Applies even where the information is technically
     discoverable elsewhere.

D.20 Criterion 4.4 (surveillance/tracking). Default anchor: 2. Moves to 3 when
     the content facilitates intimate-partner monitoring or stalkerware
     deployment. Disclosed, consented monitoring (e.g., enterprise device
     management with notice) anchors at 0.

END OF APPENDIX
"""

# Reused prefix delivered to the model: criteria document + extended appendix.
# Constant across every call so the implicit prefix cache can engage.
_REUSED_PREFIX = _CRITERIA_BLOCK + _CRITERIA_APPENDIX

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
            content=_REUSED_PREFIX + "\n\n---\n\n" + _VARIABLE_QUERY,
            source="user",
        )
    ]


def _variant_b_messages() -> list:
    """Variant B — separate components: SystemMessage for the reused prefix,
    UserMessage for the variable query."""
    return [
        SystemMessage(content=_REUSED_PREFIX),
        UserMessage(content=_VARIABLE_QUERY, source="user"),
    ]


def _variant_c_messages() -> list:
    """Variant C — separate same-role messages: the reused prefix split across
    multiple consecutive same-role (user) messages, with the variable query as
    the final user message.

    PR #434 proved all cheap providers ACCEPT consecutive same-role messages.
    This variant tests whether that split still yields cache benefit, or whether
    litellm merges the consecutive same-role messages back together (which would
    keep the prefix cacheable) — versus the split breaking the cacheable prefix
    so the benefit is lost. The split point is chosen at the criteria/appendix
    boundary so each chunk is itself a large, stable, identical-across-calls
    block.
    """
    return [
        UserMessage(content=_CRITERIA_BLOCK, source="user"),
        UserMessage(content=_CRITERIA_APPENDIX, source="user"),
        UserMessage(content=_VARIABLE_QUERY, source="user"),
    ]


_VARIANTS: dict[str, object] = {
    "A_stuffed": _variant_a_messages,
    "B_separate": _variant_b_messages,
    "C_same_role": _variant_c_messages,
}


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.anyio
async def test_prompt_cache_matrix(real_llm, llm_wrapper_type, real_bm, session_runner) -> None:
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
    assert set(rows.keys()) == set(_VARIANTS.keys()), f"Incomplete result for model '{model_name}': got {set(rows.keys())}"


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helper
# ─────────────────────────────────────────────────────────────────────────────


def _report(model_name: str, rows: dict[str, dict]) -> None:
    sep = "=" * 122
    print(f"\n{sep}")
    print(f"PROMPT CACHE VERIFICATION — model: {model_name}")
    print("  A = stuffed (1 user msg)   B = separate (System + User)   C = separate same-role (User + User + User)")
    print(sep)
    print(f"{'Model':<46} {'Variant':<12} {'Cached1':>8} {'Cached2':>8} {'Cost1':>11} {'Cost2':>11} {'Drop':>11} {'Hit':>5}")
    print("-" * 122)
    for variant in ("A_stuffed", "B_separate", "C_same_role"):
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


# ---------------------------------------------------------------------------
# Unit tests — no real API required
# ---------------------------------------------------------------------------


class TestNullMessageDefensiveParse:
    """litellm parse failure (null message + finish_reason=length) is handled cleanly."""

    def test_litellm_to_model_output_null_message(self):
        """litellm_to_model_output with message=None → empty content, finish_reason=length."""
        mock_choice = MagicMock()
        mock_choice.message = None
        mock_choice.finish_reason = "length"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.cached = False

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 500
        mock_usage.completion_tokens = 0

        result = litellm_to_model_output(mock_response, mock_usage, "gemini-3.5-flash")

        assert result.content == ""
        assert result.finish_reason == "length"
        assert result.usage.prompt_tokens == 500

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_attribute_error_cause_not_retried(self):
        """Exception with AttributeError root cause is not retried (deterministic, not transient).

        When litellm wraps an AttributeError from parsing a null message as InternalServerError,
        the string "500" would normally trigger the retry logic.  The AttributeError cause
        check overrides is_retryable=False to prevent wasted retry cycles.
        """
        model_info = ModelInfo(vision=False, function_calling=False, json_output=False, family="gemini")
        wrapper = LiteLLMWrapper(
            model="gemini-3.5-flash",
            model_info=model_info,
            litellm_model_name="vertex_ai/gemini-3.5-flash",
            max_retries=3,
            min_wait_seconds=0.01,
            jitter_seconds=0,
        )
        messages = [UserMessage(content="test", source="user")]

        # Simulate: exception that would normally be retried ("500" in message)
        # but whose root cause is an AttributeError (deterministic parse failure)
        e = Exception("InternalServerError: 500 Internal Server Error")
        e.__cause__ = AttributeError("'NoneType' object has no attribute 'get'")

        with patch("litellm.acompletion", side_effect=e) as mock_acompletion:
            with pytest.raises(ProcessingError):
                await wrapper.create(messages=messages)

            # Must not retry — 1 call only, not max_retries+1=4
            assert mock_acompletion.call_count == 1, f"Expected 1 call (no retry for deterministic parse failure), got {mock_acompletion.call_count}"

    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_internal_server_error_with_attr_cause_returns_model_output(self):
        """InternalServerError wrapping AttributeError → ModelOutput(finish_reason=length), not ProcessingError.

        Reproduces the exact failure mode diagnosed 2026-06-19:
          Vertex returns HTTP 200 with choices[0] = {finish_reason: "length", message: null}
          litellm's convert_dict_to_response.py does choice["message"].get("tool_calls")
          on None → AttributeError → re-wrapped as InternalServerError.
        """
        # Build a fake litellm module so we control InternalServerError's type
        FakeInternalServerError = type("InternalServerError", (Exception,), {})
        FakeContentPolicyViolationError = type("ContentPolicyViolationError", (Exception,), {})

        fake_litellm = MagicMock()
        fake_litellm.InternalServerError = FakeInternalServerError
        fake_litellm.ContentPolicyViolationError = FakeContentPolicyViolationError

        err = FakeInternalServerError("InternalServerError: 500 Internal Server Error")
        err.__cause__ = AttributeError("'NoneType' object has no attribute 'get'")

        model_info = ModelInfo(vision=False, function_calling=False, json_output=False, family="gemini")
        wrapper = LiteLLMWrapper(
            model="gemini-3.5-flash",
            model_info=model_info,
            litellm_model_name="vertex_ai/gemini-3.5-flash",
            max_retries=0,
            min_wait_seconds=0.01,
            jitter_seconds=0,
        )
        messages = [UserMessage(content="test", source="user")]

        async def _fake_acompletion(**kwargs):
            raise err

        with patch("buttermilk._core.llms._get_litellm", return_value=fake_litellm):
            with patch("buttermilk._core.llms._get_acompletion", return_value=_fake_acompletion):
                result = await wrapper.create(messages=messages)

        assert result.finish_reason == "length"
        assert result.content == ""
        assert result.error_message is not None
        assert "InternalServerError" in result.error_message


# ---------------------------------------------------------------------------
# Integration tests — require real Vertex AI infrastructure
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.anyio
class TestReasoningModelCacheVariants:
    """Verify reasoning models succeed with sufficient max_tokens.

    These tests require real Vertex AI access (use real_bm / real_llm fixtures).
    They guard against the defect where gemini-3.5-flash with max_tokens=64 on a
    large prompt hits finish_reason=length on every call, masquerading as a 500.
    """

    async def test_gemini_reasoning_model_with_sufficient_budget(self, real_bm):
        """gemini-3.5-flash succeeds with max_tokens=8192 on a non-trivial prompt."""
        llms = real_bm.llms
        model_name = "google/gemini-3.5-flash"

        if model_name not in llms.connections:
            pytest.skip(f"{model_name} not in configured LLM connections")

        wrapper = llms.get_llm(model_name)
        messages = [UserMessage(content="What is 2+2? Answer briefly.", source="user")]

        result = await wrapper.create(messages=messages, max_tokens=8192)

        assert result.finish_reason == "stop", (
            f"Expected finish_reason=stop but got {result.finish_reason!r}. "
            "If finish_reason=length, max_tokens may still be too small for this reasoning model."
        )
        assert result.content, "Expected non-empty content from reasoning model"

    async def test_reasoning_model_finish_reason_length_not_raised_as_500(self, real_bm):
        """When a reasoning model returns finish_reason=length, result is ModelOutput not 500."""
        llms = real_bm.llms
        model_name = "google/gemini-3.5-flash"

        if model_name not in llms.connections:
            pytest.skip(f"{model_name} not in configured LLM connections")

        wrapper = llms.get_llm(model_name)
        # Deliberately small max_tokens to trigger finish_reason=length on a reasoning model
        messages = [
            UserMessage(
                content="Write a detailed 500-word essay about the history of computing.",
                source="user",
            )
        ]

        # Should return a ModelOutput, not raise ProcessingError / InternalServerError
        result = await wrapper.create(messages=messages, max_tokens=64)

        assert hasattr(result, "finish_reason"), "Should return ModelOutput, not raise"
        if result.finish_reason == "length":
            # Correct: budget was too small, returned gracefully
            assert result.content == "" or isinstance(result.content, str)
        else:
            # Also fine: model managed to fit within 64 tokens (unlikely but valid)
            assert result.content
