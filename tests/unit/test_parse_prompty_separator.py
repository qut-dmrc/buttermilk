"""Test that _parse_prompty() raises ProcessingError when templates contain ambiguous --- markers.

This tests the fail-fast principle: Templates with horizontal rules (---) in the body
that could be misinterpreted as Prompty frontmatter delimiters should raise
ProcessingError instead of silently stripping content or producing malformed output.

The bug: _parse_prompty() uses a regex pattern that matches ANY occurrence of
`---` in the template, even when those markers are part of the body content
rather than frontmatter delimiters.

Expected behavior:
- Templates with --- pairs in the body → ProcessingError (ambiguous format)
- Templates with valid frontmatter only → parse correctly
- Templates with no frontmatter → return as-is
"""

import pytest

from buttermilk._core.exceptions import ProcessingError
from buttermilk.utils.templating import _parse_prompty


def test_parse_prompty_raises_on_horizontal_rule_in_body_no_frontmatter():
    """Test that templates with --- in body (no frontmatter) raise ProcessingError.

    Template format:
    ```
    system: You are helpful
    ---
    user: What is this?
    ```

    The --- here is a horizontal rule in the body, NOT a frontmatter delimiter.
    _parse_prompty() should detect this ambiguity and raise ProcessingError.
    """
    template_with_horizontal_rule = """system: You are a helpful assistant.

---

user: What is the meaning of this horizontal rule?"""

    # This should raise ProcessingError because --- appears in body
    # without proper frontmatter structure
    with pytest.raises(
        ProcessingError,
        match="horizontal rule|ambiguous|frontmatter|--- marker",
    ):
        _parse_prompty(template_with_horizontal_rule)


def test_parse_prompty_raises_on_horizontal_rule_after_valid_frontmatter():
    """Test that templates with valid frontmatter AND --- in body raise ProcessingError.

    Template format:
    ```
    ---
    name: test
    ---
    system: You are helpful
    ---
    user: What is this?
    ```

    The first --- pair is valid frontmatter.
    The third --- is a horizontal rule in the body content.
    This ambiguous structure should raise ProcessingError.
    """
    template_with_frontmatter_and_horizontal_rule = """---
name: test_template
description: A test
---
system: You are a helpful assistant.

---

user: What is the meaning of this horizontal rule?"""

    # This should raise ProcessingError because there's a second --- pair
    # in the body content after the frontmatter
    with pytest.raises(
        ProcessingError,
        match="horizontal rule|ambiguous|frontmatter|--- marker",
    ):
        _parse_prompty(template_with_frontmatter_and_horizontal_rule)


def test_parse_prompty_succeeds_with_valid_frontmatter_only():
    """Test that templates with ONLY valid frontmatter parse correctly.

    This is the baseline - valid frontmatter should work.
    """
    template_with_valid_frontmatter = """---
name: test_template
description: A test
---
system: You are a helpful assistant.

user: Hello!"""

    result = _parse_prompty(template_with_valid_frontmatter)

    # Should return content after frontmatter
    assert "system: You are a helpful assistant." in result
    assert "user: Hello!" in result
    # Should NOT include frontmatter
    assert "name: test_template" not in result
    assert "description: A test" not in result


def test_parse_prompty_succeeds_with_no_frontmatter():
    """Test that templates with no frontmatter return as-is.

    This is the baseline - templates without frontmatter should work.
    """
    template_without_frontmatter = """system: You are a helpful assistant.

user: Hello!"""

    result = _parse_prompty(template_without_frontmatter)

    # Should return original content unchanged
    assert result == template_without_frontmatter


def test_parse_prompty_multiple_horizontal_rules_in_body():
    """Test that multiple --- markers in body content raise ProcessingError.

    Template with multiple horizontal rules for visual separation:
    ```
    system: Instructions
    ---
    Section 1
    ---
    Section 2
    ```
    """
    template_with_multiple_horizontal_rules = """system: You are a helpful assistant.

---

user: First question

---

user: Second question"""

    # Multiple --- markers create extreme ambiguity
    # This should definitely raise ProcessingError
    with pytest.raises(
        ProcessingError,
        match="horizontal rule|ambiguous|frontmatter|--- marker",
    ):
        _parse_prompty(template_with_multiple_horizontal_rules)
