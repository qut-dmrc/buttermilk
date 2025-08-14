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
    citator = Citator(
        parameters={
            "model": "gemini-1.5-flash-latest"
        }
    )

    # Check that agent_id is generated (not empty)
    assert citator.agent_id
    assert citator.parameters["template"] == "citator"
    assert citator.parameters["fail_on_unfilled_parameters"] is True


def test_formatted_citation_model():
    """Test FormattedCitation model creation."""
    citation = FormattedCitation(
        text="Smith, J. (2023). Example Article. Journal of Examples, 1(1), 1-10.", style="APA"
    )

    assert citation.text == "Smith, J. (2023). Example Article. Journal of Examples, 1(1), 1-10."
    assert citation.style == "APA"
    assert citation.error is None

    # Test with error
    citation_with_error = FormattedCitation(text="", style="Unknown", error="Failed to generate citation")
    assert citation_with_error.error == "Failed to generate citation"


def test_citator_process_signature():
    """Test that Citator.process has the correct signature."""
    import inspect

    from buttermilk.tools.citator import Citator

    citator = Citator(parameters={"model": "gemini-1.5-flash-latest"})

    # Check that process method has the expected signature
    sig = inspect.signature(citator.process)
    params = list(sig.parameters.keys())

    # Note: 'self' is not included when inspecting bound methods
    assert "item" in params
    assert len(params) == 1  # Only 'item' parameter, no context needed
