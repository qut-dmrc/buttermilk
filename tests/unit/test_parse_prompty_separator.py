"""Test that _parse_prompty() allows horizontal rules (---) in template bodies.

Templates with horizontal rules (---) in the body should be parsed correctly,
treating the --- as part of the body content unless it is a valid frontmatter delimiter.

Previously, this raised ProcessingError due to ambiguity, but we now allow it.
"""

import pytest

from buttermilk._core.exceptions import ProcessingError
from buttermilk.utils.templating import _parse_prompty


def test_parse_prompty_allows_horizontal_rule_in_body_no_frontmatter():
    """Test that templates with --- in body (no frontmatter) are returned as-is.

    Template format:
    ```
    system: You are helpful
    ---
    user: What is this?
    ```

    The --- here is a horizontal rule in the body.
    """
    template_with_horizontal_rule = """system: You are a helpful assistant.

---

user: What is the meaning of this horizontal rule?"""

    result = _parse_prompty(template_with_horizontal_rule)
    assert result == template_with_horizontal_rule


def test_parse_prompty_allows_horizontal_rule_after_valid_frontmatter():
    """Test that templates with valid frontmatter AND --- in body work correctly.

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
    This should be parsed, returning the body with the --- intact.
    """
    template_with_frontmatter_and_horizontal_rule = """---
name: test_template
description: A test
---
system: You are a helpful assistant.

---

user: What is the meaning of this horizontal rule?"""

    expected_body = """system: You are a helpful assistant.

---

user: What is the meaning of this horizontal rule?"""

    result = _parse_prompty(template_with_frontmatter_and_horizontal_rule)
    assert result == expected_body


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


def test_parse_prompty_allows_multiple_horizontal_rules_in_body():
    """Test that multiple --- markers in body content are allowed.

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

    result = _parse_prompty(template_with_multiple_horizontal_rules)
    assert result == template_with_multiple_horizontal_rules
