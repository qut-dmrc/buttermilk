

import pytest
from unittest.mock import patch
from buttermilk.flows.helpers import get_templates, TEMPLATE_PATHS

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
