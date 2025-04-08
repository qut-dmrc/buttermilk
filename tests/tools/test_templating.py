from json import JSONDecodeError

import pytest

from buttermilk.utils.json_parser import ChatParser
from buttermilk.utils.templating import load_template
from buttermilk.utils.utils import read_json


def test_template_synth():
    flow_data = read_json("tests/data/template_synth_01.json")
    params = {
        "template": "synthesise",
        "instructions": "Carefully apply EACH of the CRITERIA in order and provide a COMPLETE and SPECIFIC explanation about whether the particular rule has been violated and how. Use quotes from the content where necessary to support your analysis.",
        "criteria": "criteria_ordinary",
        "formatting": "json_rules",
    }
    rendered, unfilled = load_template(
        parameters=params,
        untrusted_inputs=flow_data,
    )
    assert not unfilled
    assert "RULE 1, TARGETS A MARGINALIZED GROUP" in rendered
    assert "Prompt is a jinja2 template that generates prompt for LLM" not in rendered
    assert "This phrase is highly ambiguous" in rendered


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


def test_parse_invalid_json():
    parser = ChatParser(on_error="ignore")
    invalid_json = "{ This is not valid JSON }"

    result = parser.parse(invalid_json)

    assert isinstance(result, dict)
    assert "error" in result
    assert "response" in result
    assert result["error"] == "Unable to decode JSON in result"


def test_parse_raises_error():
    parser = ChatParser(on_error="raise")
    invalid_json = "{ This is not valid JSON }"

    with pytest.raises(JSONDecodeError):
        parser.parse(invalid_json)
