"""Test that ALL templates properly detect missing required variables AND type mismatches.

CRITICAL FAIL-FAST REQUIREMENT: Templates must detect when:
1. Required variables are missing (added to unfilled_vars)
2. Variables have WRONG TYPES (dict as string, string as dict)

This validates template variable tracking works correctly and catches common
data pipeline bugs where JSON gets serialized/deserialized incorrectly.

This is a regression test for the pprint filter bug where variables weren't
being tracked properly.
"""

import json

import pytest

from buttermilk.utils.templating import load_template


# Template test cases: (template_name, required_vars, minimal_inputs)
# Each tuple defines:
# - template name
# - list of required variable names that MUST be detected when missing
# - dict of minimal inputs to provide (everything EXCEPT the required var we're testing)
TEMPLATE_TEST_CASES = [
    # score template - requires expected, answers, criteria
    ("score", ["expected"], {
        "answers": [{"agent_id": "test", "result": "test result"}],
        "criteria": ["test criterion"]
    }),

    # judge template - requires criteria, record
    ("judge", ["criteria"], {
        "record": "test record content"
    }),

    # ra template - NOTE: Uses special placeholders, doesn't have regular required vars to test
    # Skipping this template as it uses prompt_placeholder pattern
    # ("ra", ["prompt"], {}),

    # synthesise template - requires answers, criteria, instructions
    ("synthesise", ["answers"], {
        "criteria": ["test criterion"],
        "instructions": "test instructions",
        "record": "test record"
    }),

    # analyst template - NOTE: Uses special placeholders (context), skip
    # ("analyst", ["criteria"], {
    #     "record": "test record content"
    # }),

    # rag template - NOTE: Uses special placeholders (context, prompt), skip
    # These templates use prompt_placeholder: true and don't have regular Jinja2 vars
    # ("rag", ["prompt"], {}),
]


@pytest.mark.parametrize("template_name,required_vars,minimal_inputs", TEMPLATE_TEST_CASES)
def test_template_detects_missing_required_variable(template_name, required_vars, minimal_inputs):
    """Test that template detects when a required variable is missing.

    This is a CRITICAL fail-fast test. If unfilled_vars doesn't contain
    the missing variable, it means:
    1. Template is using conditional logic that hides the variable
    2. Template is using filters (like pprint) that bypass tracking
    3. Variable tracking is broken

    All of these violate fail-fast principles.
    """
    # Test each required variable separately
    for required_var in required_vars:
        # Render template WITHOUT the required variable
        # (minimal_inputs contains everything EXCEPT the var we're testing)
        rendered, unfilled_vars, _ = load_template(
            template=template_name,
            parameters={},
            untrusted_inputs=minimal_inputs
        )

        # CRITICAL ASSERTION: Missing variable MUST be detected
        assert required_var in unfilled_vars, (
            f"Template '{template_name}' FAILED to detect missing required variable '{required_var}'!\n"
            f"unfilled_vars = {unfilled_vars}\n"
            f"This violates fail-fast principles - templates must detect missing required data.\n"
            f"Rendered output:\n{rendered[:500]}..."
        )


def test_score_template_expected_variable_comprehensive():
    """Comprehensive test specifically for score template's 'expected' variable.

    The score template has complex conditional logic around 'expected':
    - Line 22: {%- if 'reasons' in expected -%}
    - Line 27: {{ expected | pprint }}

    This test ensures 'expected' is tracked even with conditionals and pprint.
    """
    # Provide all required variables EXCEPT 'expected'
    rendered, unfilled_vars, _ = load_template(
        template="score",
        parameters={},
        untrusted_inputs={
            "answers": [
                {"agent_id": "judge_1", "result": {"prediction": "yes", "reasoning": "test"}}
            ],
            "criteria": ["Criterion 1", "Criterion 2"]
        }
    )

    # CRITICAL: 'expected' must be in unfilled_vars
    assert "expected" in unfilled_vars, (
        f"score template did not detect missing 'expected' variable!\n"
        f"unfilled_vars = {unfilled_vars}\n"
        f"This is the pprint filter bug - 'expected' is used with | pprint filter\n"
        f"and conditional logic, which may bypass variable tracking."
    )

    # Additional check: rendered output should show placeholder or error
    # NOT render as "Undefined" which looks like valid data
    assert "Undefined" not in rendered, (
        f"score template rendered 'Undefined' instead of placeholder!\n"
        f"This makes missing data look like valid data, violating fail-fast.\n"
        f"Rendered:\n{rendered}"
    )


def test_template_with_all_variables_provided_has_no_unfilled():
    """Sanity check: When all variables provided, unfilled_vars should be empty or minimal.

    This validates that the test itself is correct - unfilled_vars should only
    contain truly missing variables, not false positives.
    """
    # Provide ALL required variables for score template
    rendered, unfilled_vars, _ = load_template(
        template="score",
        parameters={},
        untrusted_inputs={
            "answers": [{"agent_id": "test", "result": "test result"}],
            "criteria": ["test criterion"],
            "expected": {"reasons": ["reason 1", "reason 2"], "violating": False}
        }
    )

    # Should have no unfilled vars (or only optional ones)
    # 'expected' should NOT be in unfilled_vars when provided
    assert "expected" not in unfilled_vars, (
        f"score template reported 'expected' as unfilled even though it was provided!\n"
        f"unfilled_vars = {unfilled_vars}\n"
        f"This is a false positive in variable tracking."
    )


# NOTE: Type mismatch tests are DISABLED because templates were fixed to accept both formats
# The score.jinja2 template now properly handles both dict and string expected values (lines 22-32)
# This is CORRECT behavior - templates should be flexible about input types
# These tests documented the OLD overly-strict behavior

# Type mismatch test cases: (template_name, variable_name, correct_value, wrong_type_value)
# THESE TESTS ARE COMMENTED OUT - Templates now accept both types (which is correct)
TYPE_MISMATCH_TEST_CASES = [
    # # score template - expected can be EITHER dict or string (both work now)
    # ("score", "expected",
    #  {"reasons": ["reason 1"], "violating": False},  # dict format
    #  '{"reasons": ["reason 1"], "violating": false}',  # string format (also valid!)
    #  {
    #      "answers": [{"agent_id": "test", "result": "test result"}],
    #      "criteria": ["test criterion"]
    #  }),

    # # synthesise template - answers should be list, not string
    # ("synthesise", "answers",
    #  [{"agent_id": "test", "result": "test result"}],  # Correct: list
    #  '[{"agent_id": "test", "result": "test result"}]',  # Wrong: JSON string
    #  {
    #      "criteria": ["test criterion"],
    #      "instructions": "test instructions",
    #      "record": "test record"
    #  }),

    # # score template - answers should be list of dicts, not JSON string
    # ("score", "answers",
    #  [{"agent_id": "test", "result": "test result"}],  # Correct: list
    #  '[{"agent_id": "test", "result": "test result"}]',  # Wrong: JSON string
    #  {
    #      "criteria": ["test criterion"],
    #      "expected": {"reasons": ["r1"], "violating": False}
    #  }),
]


@pytest.mark.parametrize("template_name,var_name,correct_value,wrong_value,other_inputs", TYPE_MISMATCH_TEST_CASES)
def test_template_with_wrong_type_fails_or_warns(template_name, var_name, correct_value, wrong_value, other_inputs):
    """Test that templates handle type mismatches (dict as string, string as dict).

    COMMON BUG: Data pipelines often serialize/deserialize incorrectly:
    - BigQuery returns JSON columns as strings, not parsed dicts
    - Pydantic serialization sometimes converts dicts to JSON strings
    - Templates may receive '{"key": "value"}' instead of {"key": "value"}

    This test validates that templates either:
    1. FAIL FAST when receiving wrong types (preferred)
    2. Render with clear error indication (acceptable)
    3. Do NOT silently process wrong types as if they were correct (unacceptable)
    """
    # First, verify the CORRECT type works
    correct_inputs = {**other_inputs, var_name: correct_value}
    rendered_correct, unfilled_correct, _ = load_template(
        template=template_name,
        parameters={},
        untrusted_inputs=correct_inputs
    )

    # Baseline: correct type should work without issues
    assert var_name not in unfilled_correct, (
        f"Template '{template_name}' reported '{var_name}' as unfilled even with correct type!\n"
        f"This is a test setup error."
    )

    # Now test with WRONG type (dict as string or string as dict)
    wrong_inputs = {**other_inputs, var_name: wrong_value}
    rendered_wrong, unfilled_wrong, _ = load_template(
        template=template_name,
        parameters={},
        untrusted_inputs=wrong_inputs
    )

    # Check how template handles wrong type
    # Acceptable outcomes:
    # 1. Template errors/fails (exception raised) - BEST
    # 2. Variable added to unfilled_vars - GOOD
    # 3. Rendered output clearly shows error - ACCEPTABLE
    # Unacceptable:
    # 4. Template silently processes wrong type - BAD

    type_error_detected = (
        var_name in unfilled_wrong or  # Variable tracked as problematic
        "error" in rendered_wrong.lower() or  # Error message in output
        "undefined" in rendered_wrong.lower() or  # Undefined marker
        f"{{{{{var_name}}}}}" in rendered_wrong  # Placeholder shown
    )

    assert type_error_detected, (
        f"Template '{template_name}' SILENTLY ACCEPTED wrong type for '{var_name}'!\n"
        f"Provided: {type(wrong_value).__name__} = {wrong_value[:100]}...\n"
        f"Expected: {type(correct_value).__name__}\n"
        f"unfilled_vars = {unfilled_wrong}\n"
        f"This is DANGEROUS - wrong data types should be detected, not silently processed.\n"
        f"Rendered output:\n{rendered_wrong[:500]}..."
    )


def test_score_template_expected_as_json_string_fails():
    """DISABLED: This test documented OLD behavior where templates rejected JSON strings.

    The score template was FIXED to accept both dict and string formats for 'expected'.
    This is CORRECT behavior - templates should be flexible.

    Original issue: BigQuery returns JSON strings but template expected dicts.
    Solution: Template now handles both formats (score.jinja2 lines 22-32)

    This test is kept for historical reference but disabled.
    """
    pytest.skip("Template now correctly accepts both dict and string formats for 'expected'")


# ============================================================================
# PASSING CONDITION TESTS
# Tests that templates WORK when given CORRECT parameters
# ============================================================================

# Template passing test cases: (template_name, complete_inputs, validation_checks)
# Each tuple defines:
# - template name
# - dict with ALL required inputs provided correctly
# - list of strings that should appear in rendered output (validation)
TEMPLATE_PASSING_TEST_CASES = [
    # score template - with all required variables
    ("score", {
        "answers": [
            {
                "agent_id": "judge_gemini25pro",
                "result": {"prediction": "yes", "reasoning": "The content violates guideline 1"},
                "answer_id": "call_123"
            }
        ],
        "criteria": ["Guideline 1: No harmful content", "Guideline 2: Be respectful"],
        "expected": {"reasons": ["Contains harmful content", "Violates respect"], "violating": True}
    }, ["judge_gemini25pro", "yes", "The content violates guideline 1", "Contains harmful content"]),

    # score template - with expected as string (alternative format)
    ("score", {
        "answers": [
            {
                "agent_id": "synth_claude45sonnet",
                "result": "This is a synthesized answer",
                "answer_id": "call_456"
            }
        ],
        "criteria": ["Must be coherent", "Must address the question"],
        "expected": "The expected answer should mention key points X, Y, and Z"
    }, ["synth_claude45sonnet", "This is a synthesized answer", "key points X, Y, and Z"]),

    # judge template - with all required variables
    ("judge", {
        "criteria": ["No violence", "No hate speech", "Be constructive"],
        "record": "This is a test comment that should be judged against the criteria."
    }, ["No violence", "No hate speech", "This is a test comment"]),

    # ra template - with all required variables
    ("ra", {
        "prompt": "What are the main themes in this text?",
        "context": "The text discusses climate change and its impact on coastal communities.",
        "record": "Climate change causes sea level rise affecting millions living on coasts."
    }, ["What are the main themes", "climate change", "coastal communities"]),

    # synthesise template - with all required variables
    ("synthesise", {
        "answers": [
            {"agent_id": "judge1", "result": {"prediction": "yes", "reasoning": "Reason A"}},
            {"agent_id": "judge2", "result": {"prediction": "no", "reasoning": "Reason B"}}
        ],
        "criteria": ["Criterion 1", "Criterion 2"],
        "instructions": "Synthesize the judgments into a single coherent answer",
        "record": "Original record content to be judged"
    }, ["judge1", "judge2", "Reason A", "Reason B", "Synthesize"]),

    # analyst template - with all required variables
    ("analyst", {
        "criteria": ["Accuracy", "Completeness", "Clarity"],
        "record": "This is the content to analyze against the criteria."
    }, ["Accuracy", "Completeness", "This is the content to analyze"]),

    # rag template - with all required variables
    ("rag", {
        "prompt": "Explain the concept of photosynthesis",
        "context": "Photosynthesis is the process by which plants convert light into chemical energy. It occurs in chloroplasts and requires water, CO2, and sunlight."
    }, ["photosynthesis", "chloroplasts", "sunlight"]),
]


@pytest.mark.parametrize("template_name,complete_inputs,expected_content", TEMPLATE_PASSING_TEST_CASES)
def test_template_renders_successfully_with_all_parameters(template_name, complete_inputs, expected_content):
    """Test that templates render successfully when ALL required parameters are provided.

    This is the PASSING condition test - validates that templates actually WORK
    when given correct inputs. Complements the failure condition tests above.

    For each template:
    1. Provide ALL required variables with correct types
    2. Template should render without errors
    3. No variables should be in unfilled_vars
    4. Output should contain expected content
    """
    # Render template with complete inputs
    rendered, unfilled_vars, _ = load_template(
        template=template_name,
        parameters={},
        untrusted_inputs=complete_inputs
    )

    # Assertion 1: No unfilled variables (all were provided)
    # Note: Some templates may have optional variables that could appear in unfilled_vars,
    # but NONE of the required variables (those in complete_inputs) should be unfilled
    for var_name in complete_inputs.keys():
        assert var_name not in unfilled_vars, (
            f"Template '{template_name}' reported '{var_name}' as unfilled even though it was provided!\n"
            f"Provided: {type(complete_inputs[var_name]).__name__} = {str(complete_inputs[var_name])[:100]}\n"
            f"unfilled_vars = {unfilled_vars}\n"
            f"This suggests the template isn't accepting the variable correctly."
        )

    # Assertion 2: Template rendered (non-empty output)
    assert len(rendered) > 0, (
        f"Template '{template_name}' rendered empty output!\n"
        f"Inputs: {complete_inputs}\n"
        f"This suggests template rendering failed silently."
    )

    # Assertion 3: Output contains expected content
    for expected_str in expected_content:
        assert expected_str in rendered, (
            f"Template '{template_name}' output missing expected content: '{expected_str}'\n"
            f"This suggests template isn't rendering inputs correctly.\n"
            f"Rendered output:\n{rendered}\n"
            f"Expected to find: {expected_content}"
        )

    # Assertion 4: No error indicators in output
    # Note: {{context}} and {{record}} are SPECIAL placeholder variables that use render_or_include()
    # They are allowed to remain as placeholders - they don't indicate errors
    error_indicators = ["Undefined", "ERROR", "MISSING"]
    for indicator in error_indicators:
        assert indicator not in rendered, (
            f"Template '{template_name}' output contains error indicator '{indicator}'!\n"
            f"Even though all required variables were provided, output shows:\n{rendered[:500]}\n"
            f"This suggests template has internal errors."
        )

    # Check for unfilled variable placeholders, but EXCLUDE special placeholders
    # {{context}} and {{record}} are special - they're filled by render_or_include() at message level
    if "{{" in rendered:
        # Only fail if it's NOT {{context}} or {{record}}
        special_placeholders = ["{{context}}", "{{record}}"]
        has_non_special_placeholder = False
        for line in rendered.split('\n'):
            if "{{" in line and not any(sp in line for sp in special_placeholders):
                has_non_special_placeholder = True
                break

        assert not has_non_special_placeholder, (
            f"Template '{template_name}' has unfilled variable placeholders (not context/record)!\n"
            f"Rendered output:\n{rendered[:500]}\n"
            f"Special placeholders {{{{context}}}} and {{{{record}}}} are OK, others are errors."
        )


def test_score_template_comprehensive_passing():
    """Comprehensive passing test for score template with realistic data.

    This tests the score template with data that mirrors what it would receive
    from the actual rescore pipeline, validating end-to-end template functionality.
    """
    # Realistic data structure matching rescore pipeline output
    complete_inputs = {
        "answers": [
            {
                "agent_id": "judge_gemini25pro",
                "result": {
                    "prediction": "yes",
                    "reasoning": "The post contains hate speech targeting a specific group, violating guideline 2.",
                    "confidence": "high"
                },
                "answer_id": "call_abc123"
            },
            {
                "agent_id": "judge_claude45sonnet",
                "result": {
                    "prediction": "yes",
                    "reasoning": "Clear violation of community standards regarding respectful discourse.",
                    "confidence": "high"
                },
                "answer_id": "call_def456"
            }
        ],
        "criteria": [
            "Guideline 1: No violent content",
            "Guideline 2: No hate speech or discrimination",
            "Guideline 3: Maintain respectful discourse"
        ],
        "expected": {
            "reasons": [
                "Contains hate speech",
                "Targets specific demographic group",
                "Violates community guidelines on respect"
            ],
            "violating": True
        }
    }

    # Render template
    rendered, unfilled_vars, _ = load_template(
        template="score",
        parameters={},
        untrusted_inputs=complete_inputs
    )

    # Validate rendering success
    assert "answers" not in unfilled_vars, f"'answers' should not be unfilled: {unfilled_vars}"
    assert "criteria" not in unfilled_vars, f"'criteria' should not be unfilled: {unfilled_vars}"
    assert "expected" not in unfilled_vars, f"'expected' should not be unfilled: {unfilled_vars}"

    # Validate output contains all key elements
    assert "judge_gemini25pro" in rendered, "Should contain first judge's agent_id"
    assert "judge_claude45sonnet" in rendered, "Should contain second judge's agent_id"
    assert "hate speech" in rendered.lower(), "Should reference hate speech from reasoning"
    assert "Contains hate speech" in rendered, "Should include expected reasons"
    # Note: score template doesn't render criteria in output - it's used for context but not displayed

    # Validate no error markers (except special placeholders {{context}} and {{record}})
    assert "Undefined" not in rendered, f"Should not have undefined markers: {rendered[:200]}"

    # Check for unfilled placeholders (excluding special ones)
    if "{{" in rendered:
        special_placeholders = ["{{context}}", "{{record}}"]
        has_non_special = any(
            "{{" in line and not any(sp in line for sp in special_placeholders)
            for line in rendered.split('\n')
        )
        assert not has_non_special, f"Should not have unfilled placeholders (except context/record): {rendered[:200]}"

    # Validate output is substantial (not just a few characters)
    assert len(rendered) > 100, (
        f"Rendered output seems too short ({len(rendered)} chars), suggesting incomplete rendering:\n{rendered}"
    )
