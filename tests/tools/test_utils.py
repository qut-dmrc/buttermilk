import pytest
from pydantic import Base64Str

from buttermilk.utils.templating import get_templates


@pytest.mark.parametrize(
    "pattern, expected_name, min_length",
    [
        ("criteria", "criteria_ordinary", 5240),
        ("criteria", "criteria_hatefb_factorised", 9000),
        ("synth", "synthesise", 1000),
    ],
)
def test_get_templates_default_pattern(pattern, expected_name, min_length):
    # Act
    result = get_templates(pattern)

    success = False

    for filename, content in result:
        if filename == expected_name and len(content) >= min_length:
            success = True
            break
    # Assert
    assert success


def test_b64_str_validator():
    """Test Base64Str type from Pydantic.

    Note: Pydantic's Base64Str is lenient and accepts any string,
    treating it as already base64-encoded. It only validates on decoding.
    """
    # Test that valid base64 string is accepted
    valid_b64 = Base64Str("dGVzdA==")
    assert valid_b64 == "dGVzdA=="

    # Base64Str doesn't validate format on creation, only on use
    # This is Pydantic's design - it's a string annotated type
    lenient_str = Base64Str("not-actually-base64")
    assert isinstance(lenient_str, str)
