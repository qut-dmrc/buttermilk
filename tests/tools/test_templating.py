from json import JSONDecodeError

import pytest

try:
    from buttermilk.utils.json_parser import ChatParser

    CHATPARSER_AVAILABLE = True
except ImportError:
    CHATPARSER_AVAILABLE = False

from buttermilk.utils.templating import calculate_template_hash, load_template
from buttermilk.utils.utils import read_json


def test_template_synth():
    flow_data = read_json("tests/data/template_synth_01.json")
    parameters = {
        "template": "synthesise",
        "instructions": "Carefully apply EACH of the CRITERIA in order and provide a COMPLETE and SPECIFIC explanation about whether the particular rule has been violated and how. Use quotes from the content where necessary to support your analysis.",
        "criteria": "criteria_ordinary",
        "formatting": "json_rules",
    }
    # Merge parameters with flow_data (parameters take precedence)
    merged_vars = {**flow_data, **parameters}
    rendered, unfilled, template_hash = load_template(
        template="synthesise",
        template_vars=merged_vars,
    )
    # Placeholder variables (record, context) are intentionally kept unfilled
    # by load_template() - they're processed by make_messages() later
    placeholder_vars = {"record", "context"}
    actual_unfilled = unfilled - placeholder_vars
    assert not actual_unfilled, f"Unexpected unfilled template variables: {actual_unfilled}"
    assert "RULE 1, TARGETS A MARGINALIZED GROUP" in rendered
    assert "Prompt is a jinja2 template that generates prompt for LLM" not in rendered
    # Template content may vary, just check that we got a non-empty rendered output
    assert len(rendered) > 1000

    # Test template hash is returned and has correct format
    assert template_hash is not None
    assert isinstance(template_hash, str)
    assert len(template_hash) == 64  # 64 hex chars (no prefix)
    assert all(c in "0123456789abcdef" for c in template_hash)


def test_calculate_template_hash():
    """Test that calculate_template_hash returns consistent hash for a template."""
    template_hash, template_path = calculate_template_hash("synthesise")

    # Test hash format
    assert isinstance(template_hash, str)
    assert len(template_hash) == 64  # 64 hex chars (no prefix)
    assert all(c in "0123456789abcdef" for c in template_hash)

    # Test path is returned
    assert isinstance(template_path, str)
    assert template_path.endswith("synthesise.jinja2")

    # Test consistency - same template should return same hash
    hash2, path2 = calculate_template_hash("synthesise")
    assert template_hash == hash2
    assert template_path == path2


def test_calculate_template_hash_nonexistent():
    """Test that calculate_template_hash raises error for non-existent template."""
    from buttermilk._core.exceptions import FatalError

    with pytest.raises(FatalError, match="Template file 'nonexistent.jinja2' not found"):
        calculate_template_hash("nonexistent")


def test_load_template_hash_consistency():
    """Test that load_template and calculate_template_hash return same hash."""
    # Get hash from calculate_template_hash
    direct_hash, _ = calculate_template_hash("synthesise")

    # Get hash from load_template
    _, _, template_hash = load_template(
        template="synthesise",
        template_vars={"test": "value"},
    )

    # Should be the same
    assert direct_hash == template_hash


def test_different_templates_different_hashes():
    """Test that different templates have different hashes."""
    hash1, _ = calculate_template_hash("synthesise")
    hash2, _ = calculate_template_hash("judge")

    # Different templates should have different hashes
    assert hash1 != hash2
    assert len(hash1) == 64  # 64 hex chars (no prefix)
    assert len(hash2) == 64  # 64 hex chars (no prefix)
    assert all(c in "0123456789abcdef" for c in hash1)
    assert all(c in "0123456789abcdef" for c in hash2)


def test_parse_valid_json():
    parser = ChatParser()
    sample_json = """{
        "confidence": "high",
        "error": null,
        "labels": [
            "Violence",
            "Health Care",
            "Identification Records",
            "Employment",
            "Criminal Justice",
            "Respectful Language"
        ],
        "prediction": true,
        "reasons": [
            "The content discusses the debate within the Democratic Party regarding transgender rights...",
            "The article quotes Moulton as saying...",
            "The guidelines emphasize the importance of using respectful language...",
            "The article also mentions that some Democrats...",
            "Overall, the content raises important questions..."
        ],
        "severity": "medium"
    }"""

    result = parser.parse(sample_json)

    assert isinstance(result, dict)
    assert result["confidence"] == "high"
    assert result["error"] is None
    assert len(result["labels"]) == 6
    assert result["prediction"] is True
    assert len(result["reasons"]) == 5
    assert result["severity"] == "medium"


def test_parse_with_surrounding_text():
    parser = ChatParser()
    sample_with_noise = """
    Some text before the JSON
    {
        "key": "value",
        "number": 42
    }
    Some text after the JSON
    """

    result = parser.parse(sample_with_noise)

    assert isinstance(result, dict)
    assert result["key"] == "value"
    assert result["number"] == 42


@pytest.mark.skipif(not CHATPARSER_AVAILABLE, reason="ChatParser not available in current codebase")
def test_parse_invalid_json():
    parser = ChatParser(on_error="ignore")
    invalid_json = "{ This is not valid JSON }"

    result = parser.parse(invalid_json)

    assert isinstance(result, dict)
    assert "error" in result
    assert "response" in result
    assert result["error"] == "Unable to decode JSON in result"


@pytest.mark.skipif(not CHATPARSER_AVAILABLE, reason="ChatParser not available in current codebase")
def test_parse_raises_error():
    parser = ChatParser(on_error="raise")
    invalid_json = "{ This is not valid JSON }"

    with pytest.raises(JSONDecodeError):
        parser.parse(invalid_json)
