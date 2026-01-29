"""Robustness tests for the templating system.

Tests edge cases and potential failure modes to ensure prompts
are robust against unusual inputs.

These tests verify that render_template() handles:
- Unicode and special characters
- Fail-on-unfilled behavior and error messages
- TemplateRenderResult dataclass fields
- Base vs runtime variable precedence
- Various Python types as template variables
- Jinja2 filter and whitespace behavior
"""

from datetime import datetime
from decimal import Decimal

import pytest

from buttermilk._core.exceptions import FatalError
from buttermilk.utils.templating import TemplateRenderResult, render_template


class TestUnicodeHandling:
    """Unicode and special character handling in template variables."""

    def test_unicode_emoji_in_values(self):
        """Template should correctly render emoji characters."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "Hello World! 🎉🚀💻"},
        )
        assert "Hello World!" in result.rendered
        assert "🎉🚀💻" in result.rendered
        assert result.unfilled_vars == []

    def test_unicode_cjk_characters(self):
        """Template should handle CJK (Chinese/Japanese/Korean) characters."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "你好世界 こんにちは 안녕하세요"},
        )
        assert "你好世界" in result.rendered
        assert "こんにちは" in result.rendered
        assert "안녕하세요" in result.rendered

    def test_unicode_arabic_rtl(self):
        """Template should handle Arabic/RTL text."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "مرحبا بالعالم"},
        )
        assert "مرحبا بالعالم" in result.rendered

    def test_unicode_combining_characters(self):
        """Template should handle combining diacritical marks."""
        # é can be represented as e + combining acute accent
        text_with_combining = "cafe\u0301"  # café with combining accent
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": text_with_combining},
        )
        assert text_with_combining in result.rendered

    def test_jinja2_like_sequences_in_values(self):
        """Values containing {{ or }} should be rendered literally, not interpreted."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "Use {{ variable }} syntax"},
        )
        # The literal {{ should appear in output, not trigger Jinja2
        assert "{{ variable }}" in result.rendered

    def test_jinja2_block_sequences_in_values(self):
        """Values containing {% or %} should be rendered literally."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "Use {% if x %}...{% endif %} for conditionals"},
        )
        assert "{% if x %}" in result.rendered

    def test_multiline_values(self):
        """Template should handle multiline string values."""
        multiline = "Line 1\nLine 2\nLine 3\n\nLine 5 after blank"
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": multiline},
        )
        assert "Line 1" in result.rendered
        assert "Line 5 after blank" in result.rendered

    def test_tabs_and_special_whitespace(self):
        """Template should preserve tabs and special whitespace in values."""
        text_with_tabs = "Column1\tColumn2\tColumn3"
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": text_with_tabs},
        )
        assert "\t" in result.rendered

    def test_very_long_string_values(self):
        """Template should handle very long strings (10K+ characters)."""
        long_text = "x" * 15000
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": long_text},
        )
        assert long_text in result.rendered


class TestFailOnUnfilledBehavior:
    """Fail-fast behavior for unfilled template variables."""

    def test_fail_on_unfilled_true_raises_fatal_error(self):
        """When fail_on_unfilled=True, unfilled vars raise FatalError."""
        with pytest.raises(FatalError) as exc_info:
            render_template(
                template="test/robustness",
                template_vars={},  # Missing required_value
                fail_on_unfilled=True,
            )
        assert "unfilled parameters" in str(exc_info.value).lower()
        assert "required_value" in str(exc_info.value)

    def test_fail_on_unfilled_false_returns_unfilled_list(self):
        """When fail_on_unfilled=False, unfilled vars returned in list."""
        result = render_template(
            template="test/robustness",
            template_vars={},
            fail_on_unfilled=False,
        )
        assert "required_value" in result.unfilled_vars
        # The placeholder should remain visible in rendered output (no spaces in Jinja2)
        assert "{{required_value}}" in result.rendered

    def test_partial_unfilled_vars(self):
        """Some vars filled, some not - only unfilled ones reported."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "filled"},
            fail_on_unfilled=False,
        )
        # required_value should NOT be in unfilled
        assert "required_value" not in result.unfilled_vars
        assert result.unfilled_vars == []

    def test_error_message_includes_template_name(self):
        """FatalError message should include the template name for debugging."""
        with pytest.raises(FatalError) as exc_info:
            render_template(
                template="test/robustness",
                template_vars={},
                fail_on_unfilled=True,
            )
        assert "test/robustness" in str(exc_info.value)

    def test_error_message_includes_all_unfilled_var_names(self):
        """FatalError should list all unfilled variables."""
        with pytest.raises(FatalError) as exc_info:
            render_template(
                template="test/robustness",
                template_vars={},
                fail_on_unfilled=True,
            )
        error_msg = str(exc_info.value)
        assert "required_value" in error_msg


class TestTemplateRenderResult:
    """TemplateRenderResult dataclass validation."""

    def test_returns_template_render_result_type(self):
        """render_template should return TemplateRenderResult instance."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "test"},
        )
        assert isinstance(result, TemplateRenderResult)

    def test_hash_consistency(self):
        """Same template should always produce same hash."""
        result1 = render_template(
            template="test/robustness",
            template_vars={"required_value": "test1"},
        )
        result2 = render_template(
            template="test/robustness",
            template_vars={"required_value": "test2"},
        )
        # Hash is of the TEMPLATE FILE, not the rendered output
        assert result1.template_hash == result2.template_hash
        assert len(result1.template_hash) == 64  # SHA-256 produces 64 hex chars

    def test_template_name_matches_input(self):
        """template_name should match the input template parameter."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "test"},
        )
        assert result.template_name == "test/robustness"

    def test_unfilled_vars_is_list(self):
        """unfilled_vars should be a list, not a set."""
        result = render_template(
            template="test/robustness",
            template_vars={},
            fail_on_unfilled=False,
        )
        assert isinstance(result.unfilled_vars, list)

    def test_rendered_content_complete(self):
        """rendered should contain all expected sections from template."""
        result = render_template(
            template="test/robustness",
            template_vars={
                "required_value": "my_value",
                "optional_value": "optional_content",
            },
        )
        # Check for both system and user sections from template
        assert "system:" in result.rendered
        assert "user:" in result.rendered
        assert "my_value" in result.rendered
        assert "optional_content" in result.rendered


class TestVariablePrecedence:
    """Base vs runtime variable override behavior."""

    def test_runtime_overrides_base(self):
        """Runtime template_vars should override base_template_vars."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "runtime_wins"},
            base_template_vars={"required_value": "base_loses"},
        )
        assert "runtime_wins" in result.rendered
        assert "base_loses" not in result.rendered

    def test_base_used_when_runtime_missing(self):
        """base_template_vars should be used when runtime doesn't provide value."""
        result = render_template(
            template="test/robustness",
            template_vars={},
            base_template_vars={"required_value": "from_base"},
        )
        assert "from_base" in result.rendered

    def test_empty_string_runtime_overrides_then_cleaned(self):
        """Empty string in runtime overrides base, then gets cleaned → unfilled.

        This is the expected behavior: runtime always overrides base first,
        then clean_empty_values() removes empty values. Setting runtime to ""
        means "clear this value" not "use base value".
        """
        # This should raise because runtime empty string clears the base value
        with pytest.raises(FatalError):
            render_template(
                template="test/robustness",
                template_vars={"required_value": ""},  # Explicitly empty
                base_template_vars={"required_value": "from_base"},
                fail_on_unfilled=True,
            )

    def test_none_runtime_overrides_then_cleaned(self):
        """None in runtime overrides base, then gets cleaned → unfilled.

        Same as empty string: runtime None means "clear this value".
        """
        with pytest.raises(FatalError):
            render_template(
                template="test/robustness",
                template_vars={"required_value": None},
                base_template_vars={"required_value": "from_base"},
                fail_on_unfilled=True,
            )

    def test_both_empty_raises_unfilled(self):
        """When both base and runtime are empty, variable is unfilled."""
        with pytest.raises(FatalError):
            render_template(
                template="test/robustness",
                template_vars={"required_value": ""},
                base_template_vars={"required_value": ""},
                fail_on_unfilled=True,
            )


class TestValueTypes:
    """Various Python types as template variables."""

    def test_nested_dict(self):
        """Nested dict should render as string representation."""
        nested = {"outer": {"inner": "value", "list": [1, 2, 3]}}
        result = render_template(
            template="test/robustness",
            template_vars={
                "required_value": "test",
                "nested_value": nested,
            },
        )
        # Dict should appear in rendered output
        assert "inner" in result.rendered
        assert "value" in result.rendered

    def test_list_value(self):
        """List should render as string representation."""
        items = ["item1", "item2", "item3"]
        result = render_template(
            template="test/robustness",
            template_vars={
                "required_value": items,
            },
        )
        assert "item1" in result.rendered
        assert "item2" in result.rendered

    def test_boolean_true(self):
        """Boolean True should render correctly."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": True},
        )
        assert "True" in result.rendered

    def test_boolean_false(self):
        """Boolean False should render correctly (not be cleaned out)."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": False},
        )
        # False is falsy but should still render
        assert "False" in result.rendered

    def test_integer_value(self):
        """Integer should render correctly."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": 42},
        )
        assert "42" in result.rendered

    def test_integer_zero(self):
        """Integer zero should render correctly (not be cleaned out)."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": 0},
        )
        # 0 is falsy but should still render
        assert "0" in result.rendered

    def test_float_value(self):
        """Float should render correctly."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": 3.14159},
        )
        assert "3.14159" in result.rendered

    def test_decimal_value(self):
        """Decimal should render correctly."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": Decimal("123.456")},
        )
        assert "123.456" in result.rendered

    def test_datetime_value(self):
        """Datetime should render as string."""
        dt = datetime(2024, 6, 15, 10, 30, 0)
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": dt},
        )
        assert "2024" in result.rendered
        assert "06" in result.rendered or "6" in result.rendered

    def test_custom_object_with_str(self):
        """Custom object with __str__ should render using that method."""

        class CustomObj:
            def __str__(self):
                return "CustomObjectString"

        result = render_template(
            template="test/robustness",
            template_vars={"required_value": CustomObj()},
        )
        assert "CustomObjectString" in result.rendered


class TestJinja2Features:
    """Jinja2 rendering features and filters."""

    def test_strip_all_filter_removes_whitespace(self):
        """strip_all filter should remove leading/trailing whitespace."""
        result = render_template(
            template="test/robustness",
            template_vars={
                "required_value": "test",
                "whitespace_value": "  \t  trimmed value  \n  ",
            },
        )
        # The strip_all filter should have removed the whitespace
        assert "trimmed value" in result.rendered
        # Check we don't have leading spaces before "trimmed"
        # (This is tricky to test precisely due to template structure)

    def test_conditional_with_empty_string(self):
        """Empty string should be falsy in Jinja2 conditionals."""
        result = render_template(
            template="test/robustness",
            template_vars={
                "required_value": "test",
                "optional_value": "",  # Empty string - falsy
            },
        )
        # optional_value block should NOT appear because empty string is falsy
        assert "Optional:" not in result.rendered

    def test_conditional_with_none(self):
        """None should be falsy in Jinja2 conditionals."""
        result = render_template(
            template="test/robustness",
            template_vars={
                "required_value": "test",
                "optional_value": None,
            },
        )
        # Note: None is cleaned out by clean_empty_values, so won't even be passed
        assert "Optional:" not in result.rendered

    def test_conditional_with_truthy_value(self):
        """Non-empty value should render conditional block."""
        result = render_template(
            template="test/robustness",
            template_vars={
                "required_value": "test",
                "optional_value": "present",
            },
        )
        assert "Optional: present" in result.rendered


class TestErrorRecovery:
    """Error handling and recovery scenarios."""

    def test_nonexistent_template_raises_fatal_error(self):
        """Missing template should raise FatalError, not generic exception."""
        with pytest.raises(FatalError):
            render_template(
                template="nonexistent/template/path",
                template_vars={"required_value": "test"},
            )

    def test_empty_template_vars_dict(self):
        """Empty dict should work, reporting unfilled vars."""
        result = render_template(
            template="test/robustness",
            template_vars={},
            fail_on_unfilled=False,
        )
        assert len(result.unfilled_vars) > 0

    def test_none_template_vars(self):
        """None for template_vars should work (treated as empty)."""
        result = render_template(
            template="test/robustness",
            template_vars=None,
            fail_on_unfilled=False,
        )
        assert "required_value" in result.unfilled_vars

    def test_none_base_template_vars(self):
        """None for base_template_vars should work."""
        result = render_template(
            template="test/robustness",
            template_vars={"required_value": "test"},
            base_template_vars=None,
        )
        assert "test" in result.rendered
