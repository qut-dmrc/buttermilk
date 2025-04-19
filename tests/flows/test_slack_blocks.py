import pytest

from buttermilk._core.contract import AgentOutput
from buttermilk._core.types import Record
from buttermilk.agents.ui.formatting.slackblock import (
    format_response,
)


@pytest.fixture
def sample_agent_output():
    """Create a sample agent output for testing formatting."""
    return AgentOutput(
        content="This is the main content",
        outputs={
            "prediction": "Test prediction",
            "confidence": "High",
            "severity": "Medium",
            "labels": ["label1", "label2"],
            "reasons": ["First reason", "Second reason"],
            "additional_field": "Additional value",
        },
        metadata={
            "model": "gpt-4",
            "input": "Original query",
            "criteria": "Test criteria",
        },
    )


@pytest.fixture
def sample_record():
    """Create a sample record for testing."""
    return Record(
        record_id="rec123",
        title="Test Title",
        fulltext="Test full text content",
        paragraphs=["Paragraph 1", "Paragraph 2"],
        metadata={
            "title": "Metadata Title",
            "outlet": "News Outlet",
            "date": "2023-01-01",
            "url": "https://example.com",
        },
    )


def test_format_response_string():
    """Test formatting a simple string."""
    result = format_response("Simple text")
    assert result == ["Simple text"]


def test_format_response_mapping():
    """Test formatting a dictionary."""
    result = format_response({"key1": "value1", "key2": "value2"})
    assert len(result) == 2
    assert "*key1*: value1" in result[0]
    assert "*key2*: value2" in result[1]


def test_format_response_sequence():
    """Test formatting a sequence."""
    result = format_response(["item1", "item2", ""])
    assert len(result) == 2
    assert "item1" in result
    assert "item2" in result
    # Empty items should be filtered out


def test_format_response_empty():
    """Test formatting empty inputs."""
    assert format_response(None) == []
    assert format_response("") == []
