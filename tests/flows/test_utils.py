

import pytest
from pydantic import Base64Str

from buttermilk.utils.templating import get_templates


@pytest.mark.parametrize(
    "pattern, expected_name, min_length",
    [
        ("criteria", "criteria_ordinary.jinja2", 5240),
        ("criteria", "criteria_hatefb_factorised.jinja2", 9000),
        ("synth", "synthesise.jinja2", 1000),
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
    assert pytest.raises(ValueError, match="Invalid base64 string") == Base64Str(
        "invalid"
    )
    assert Base64Str("dGVzdA==").decode() == "test"

def jstest_input_map():
    pass