"""Test the Citator agent initialization and functionality."""

from buttermilk.tools.citator import Citator, FormattedCitation


def test_citator_initialization():
    """Test that Citator can be initialized with required parameters."""
    citator = Citator(parameters={"model": "gemini-1.5-flash-latest", "template": "citator"})

    # Check that agent_id is generated (not empty)
    assert citator.agent_id
    assert citator.description == "Generates a citation for a given text using an LLM."
    assert citator.parameters["model"] == "gemini-1.5-flash-latest"
    assert citator.parameters["template"] == "citator"
    assert citator.output_model == FormattedCitation


def test_citator_initialization_with_defaults():
    """Test that Citator can be initialized with minimal parameters."""
    citator = Citator(parameters={"model": "gemini-1.5-flash-latest"})

    # Check that agent_id is generated (not empty)
    assert citator.agent_id
    # Check that default template is set
    assert citator.parameters["template"] == "citator"
    # Check that output model is set
    assert citator.output_model == FormattedCitation


def test_formatted_citation_model():
    """Test FormattedCitation model creation."""
    citation = FormattedCitation(title="Example Article", citation="Smith, J. (2023). Example Article. Journal of Examples, 1(1), 1-10.", style="APA")

    assert citation.title == "Example Article"
    assert citation.citation == "Smith, J. (2023). Example Article. Journal of Examples, 1(1), 1-10."
    assert citation.style == "APA"
    assert citation.error is None

    # Test with error
    citation_with_error = FormattedCitation(title="Unknown", citation="", style="Unknown", error="Failed to generate citation")
    assert citation_with_error.error == "Failed to generate citation"


def test_citator_process_signature():
    """Test that Citator.process has the correct signature for pipeline integration."""
    import inspect
    from collections.abc import AsyncGenerator

    from buttermilk.tools.citator import Citator

    citator = Citator(parameters={"model": "gemini-1.5-flash-latest"})

    # Check that process method has the expected signature for Processor protocol
    sig = inspect.signature(citator.process)
    params = list(sig.parameters.keys())

    # Note: 'self' is not included when inspecting bound methods
    assert "item" in params
    assert "processor_stage" in params
    assert "kwargs" in params

    # Check return type is AsyncGenerator
    return_annotation = sig.return_annotation
    # The annotation is AsyncGenerator[Record, None]
    assert hasattr(return_annotation, "__origin__")
    assert return_annotation.__origin__ is AsyncGenerator
