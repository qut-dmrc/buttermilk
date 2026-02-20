"""Test that undefined variables with pprint filter fail-fast properly.

This test uses the REAL score template to demonstrate the bug where
undefined variables passed through the pprint filter render as "Undefined"
instead of showing {{variable_name}} placeholder.

CRITICAL BUG: Template should fail-fast when required variables are missing,
but instead silently renders "Undefined" which looks like valid data.
"""

from buttermilk.utils.templating import load_template


def test_score_template_with_missing_expected_shows_placeholder():
    """Test that score template with missing 'expected' variable shows {{expected}} not 'Undefined'.

    The score template uses {{ expected | pprint }} on line 27.
    When 'expected' is undefined, this should render as {{expected}},
    but currently renders as "Undefined".

    This test SHOULD FAIL, demonstrating the pprint filter bug.
    """
    # Use ACTUAL score template from buttermilk/templates/prompt/score.jinja2
    # Provide answers and criteria (required), but NOT expected (triggers bug)
    rendered, unfilled_vars, _ = load_template(
        template="score",
        template_vars={
            "answers": [{"agent_id": "test", "result": "test result"}],
            "criteria": ["test criterion"],
            # 'expected' is MISSING - this should trigger fail-fast
        },
    )

    # CRITICAL ASSERTION: unfilled_vars should detect missing 'expected'
    assert "expected" in unfilled_vars, f"Template should detect 'expected' as unfilled, but unfilled_vars = {unfilled_vars}"

    # CRITICAL ASSERTION: Should render as {{expected}} placeholder
    assert "{{expected}}" in rendered, f"Undefined 'expected' with pprint filter should render as {{{{expected}}}}, but rendered as:\n{rendered}"

    # CRITICAL ASSERTION: Should NOT render as "Undefined"
    assert "Undefined" not in rendered, f"Should not render as 'Undefined' string, but found it in:\n{rendered}"
