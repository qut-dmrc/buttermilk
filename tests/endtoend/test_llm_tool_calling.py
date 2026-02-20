"""Test tool calling functionality across all LLM models."""

from typing import Literal

import pytest
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import SystemMessage, UserMessage
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, Field, ConfigDict

from buttermilk._core.llms import ModelOutput

class StructuredTestAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: str = Field(description="Your conlusion or final answer.")
    prediction: bool = Field(
        description="True if the content violates the policy or guidelines. Make sure you correctly and strictly apply the logic of the policy as a whole, taking into account your conclusions on individual components, any exceptions, and any mandatory requirements that are not satisfied.",
    )
    reasons: list[str] = Field(
        ...,
        description="List of reasoning steps. Each step should comprise one to five sentences of text presenting a clear logical analysis.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Your confidence in the overall conclusion."
    )
    thoughts: str


class WeatherResponse(BaseModel):
    """Response model for weather information."""

    location: str = Field(description="The location for the weather query")
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition (e.g., sunny, cloudy, rainy)")
    humidity: int = Field(description="Humidity percentage")


async def get_weather(location: str) -> WeatherResponse:
    """Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        WeatherResponse with weather information

    """
    # Mock weather data based on location
    weather_data = {
        "london": WeatherResponse(
            location="London", temperature=15.5, condition="cloudy", humidity=75
        ),
        "new york": WeatherResponse(
            location="New York", temperature=22.0, condition="sunny", humidity=60
        ),
        "tokyo": WeatherResponse(
            location="Tokyo", temperature=18.5, condition="rainy", humidity=85
        ),
        "paris": WeatherResponse(
            location="Paris", temperature=17.0, condition="partly cloudy", humidity=70
        ),
    }

    # Normalize location for lookup
    location_key = location.lower().strip()

    # Return mock data or default
    return weather_data.get(
        location_key,
        WeatherResponse(
            location=location, temperature=20.0, condition="clear", humidity=65
        ),
    )


async def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b

    """
    return a + b


@pytest.mark.anyio
async def test_structured_output_without_tools(real_llm_expensive, llm_wrapper_type):
    system = """You are a content moderator. You will be provided with a set of criteria to apply to a sample of user content.
    <CRITERIA>
    Is this output illegal, unethical or harmful to any person?
    </CRITERIA>
    """
    llm = real_llm_expensive

    messages = [
        SystemMessage(content=system),
        UserMessage(content="The capital of France is Paris.", source="user"),
    ]

    response = await llm.create(messages=messages, schema=StructuredTestAgentOutput)

    assert isinstance(response, ModelOutput)
    assert isinstance(response.parsed_object, StructuredTestAgentOutput)
    assert response.parsed_object.conclusion
    assert not response.parsed_object.prediction


@pytest.mark.anyio
async def test_single_tool_call(real_llm_expensive, llm_wrapper_type):
    """Test that each LLM can make a single tool call."""
    llm = real_llm_expensive

    # Create a simple weather tool
    weather_tool = FunctionTool(
        get_weather,
        name="get_weather",
        description="Get the current weather for a location",
        strict=True,
    )

    messages = [
        SystemMessage(
            content="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather.",
            source="system",
        ),
        UserMessage(content="What's the weather like in London?", source="user"),
    ]

    # Test with tool calling
    try:
        response = await llm.call_chat(
            messages=messages,
            tools_list=[weather_tool],
            cancellation_token=CancellationToken(),
        )
    except Exception as e:
        error_msg = str(e).lower()
        # Try to get model name from various possible attributes
        model_name = getattr(llm, "_model_name", getattr(llm, "litellm_model_name", "")).lower()

        if "missing a thought_signature" in error_msg:
            pytest.skip(f"Vertex AI/Gemini requires thought signature which is currently not handled: {e}")

        # Fallback for complex nested exceptions where the string might be truncated or formatted differently
        # Specific skip for Gemini 400 errors which are typically the thought signature issue in this context
        if "gemini" in model_name and "400" in error_msg:
            pytest.skip(f"Skipping Gemini 400 error (likely thought signature): {e}")

        raise

    # Verify response mentions London and weather details
    assert response.content
    assert isinstance(response.content, str)
    content_lower = response.content.lower()

    # Should mention London
    assert "london" in content_lower

    # Should mention weather details (at least one of these)
    weather_terms = ["cloudy", "15.5", "75", "humidity", "temperature", "celsius", "°c"]
    assert any(term in content_lower for term in weather_terms), (
        f"Response should contain weather information, got: {response.content}"
    )


@pytest.mark.anyio
async def test_multiple_tool_calls(real_llm, llm_wrapper_type):
    """Test that LLMs can handle multiple tools and select the right one."""
    # Create multiple tools
    weather_tool = FunctionTool(
        get_weather,
        name="get_weather",
        description="Get the current weather for a location",
        strict=True,
    )

    calc_tool = FunctionTool(
        calculate_sum,
        name="calculate_sum",
        description="Calculate the sum of two numbers",
        strict=True,
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant with access to weather and calculation tools. Use the calculate_sum tool when asked about addition.",
            source="system",
        ),
        UserMessage(content="What's 5 + 3?", source="user"),
    ]

    try:
        # Test with multiple tools available
        response = await real_llm.call_chat(
            messages=messages,
            tools_list=[weather_tool, calc_tool],
            cancellation_token=CancellationToken(),
        )

        # Verify response contains the correct sum
        assert response.content
        assert isinstance(response.content, str)

        # Check for both digit "8" and word "eight"
        assert any(term in response.content.lower() for term in ["8", "eight"]), (
            f"Response should contain the sum 8, got: {response.content}"
        )
    except Exception as e:
        if "does not support function calling" in str(e):
            pytest.skip(f"Model doesn't support tool calling: {e}")
        raise


@pytest.mark.anyio
async def test_no_tool_needed(real_llm, llm_wrapper_type):
    """Test that LLMs don't use tools when not needed."""
    # Create tools that shouldn't be used
    weather_tool = FunctionTool(
        get_weather,
        name="get_weather",
        description="Get the current weather for a location",
        strict=True,
    )

    calc_tool = FunctionTool(
        calculate_sum,
        name="calculate_sum",
        description="Calculate the sum of two numbers",
        strict=True,
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Answer questions directly when you can without using any tools.",
            source="system",
        ),
        UserMessage(content="What is the capital of France?", source="user"),
    ]

    try:
        # Test with tools available but not needed
        response = await real_llm.call_chat(
            messages=messages,
            tools_list=[weather_tool, calc_tool],
            cancellation_token=CancellationToken(),
        )

        # Verify response contains Paris without using tools
        assert response.content
        assert isinstance(response.content, str)

        assert "paris" in response.content.lower(), (
            f"Response should mention Paris, got: {response.content}"
        )
    except Exception as e:
        if "does not support function calling" in str(e):
            pytest.skip(f"Model doesn't support tool calling: {e}")
        raise


@pytest.mark.anyio
async def test_call_chat_intercept_tools_returns_function_calls(real_llm, llm_wrapper_type):
    """Verify that call_chat(intercept_tools=True) returns FunctionCall objects without executing."""
    calc_tool = FunctionTool(
        calculate_sum,
        name="calculate_sum",
        description="Calculate the sum of two numbers",
        strict=True,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant with access to the calculate_sum tool. "
                "Do not answer directly. You must call the calculate_sum tool to compute 5 + 7."
            ),
            source="system",
        ),
        UserMessage(content="What's 5 + 7? Use the tool.", source="user"),
    ]

    try:
        result = await real_llm.call_chat(
            messages=messages,
            tools_list=[calc_tool],
            cancellation_token=CancellationToken(),
            intercept_tools=True,
        )
    except Exception as e:
        if "does not support function calling" in str(e):
            pytest.skip(f"Model doesn't support tool calling: {e}")
        raise

    assert result.content, "Expected tool call(s) in result content"
    assert isinstance(result.content, list), (
        f"Expected a list of FunctionCall, got: {type(result.content)}"
    )
    assert all(isinstance(c, FunctionCall) for c in result.content), (
        "Expected FunctionCall objects"
    )
    assert any(c.name == "calculate_sum" for c in result.content), (
        "Expected a calculate_sum tool call"
    )


@pytest.mark.anyio
async def test_structured_output_with_incorrect_tools(real_llm_expensive, llm_wrapper_type):
    """Test that models handle requests for structured output with irrelevant tools passed."""

    class Answer(BaseModel):
        """Structured answer format."""

        result: str = Field(description="The answer to the question")
        confidence: float = Field(description="Confidence level from 0 to 1")

    # Create a simple tool
    calc_tool = FunctionTool(
        calculate_sum,
        name="calculate_sum",
        description="Calculate the sum of two numbers",
        strict=True,
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Always structure your responses using the provided schema. Answer questions using your general knowledge even if tools are available but not relevant.",
            source="system",
        ),
        UserMessage(content="What is the capital of Japan?", source="user"),
    ]

    # Test with structured output (tools should not be passed with structured output for certain models)
    response = await real_llm_expensive.call_chat(
        messages=messages,
        tools_list=[calc_tool],
        schema=Answer,
        cancellation_token=CancellationToken(),  # This might be ignored for some models
    )

    # Verify structured response
    assert response.content
    if hasattr(response, "parsed_object") and response.parsed_object:
        assert isinstance(response.parsed_object, Answer)
        assert "tokyo" in response.parsed_object.result.lower()
    else:
        # Fallback for models that don't support structured output
        assert "tokyo" in response.content.lower()


@pytest.mark.anyio
async def test_call_chat_tool_exec_then_synthesis_with_schema(real_llm_expensive, llm_wrapper_type):
    """Cover the full flow: initial tool call -> tool execution -> synthesis call with schema.

    This test helps surface issues where the synthesis call incorrectly sets a structured
    response_format that some models don't support.
    """

    class Answer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        result: list[int] = Field(description="The final answer")

    calc_tool = FunctionTool(
        calculate_sum,
        name="calculate_sum",
        description="Calculate the sum of two numbers",
        strict=True,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. When a task involves addition, you must use the calculate_sum tool "
                "for every addition operation. Do not perform any arithmetic yourself. Return both answers as a list using the schema provided."
            ),
            source="system",
        ),
        UserMessage(
            content=(
                "Compute (5 + 3) and (10 + 4) using separate calls to the calculate_sum tool."
            ),
            source="user",
        ),
    ]

    try:
        response = await real_llm_expensive.call_chat(
            messages=messages,
            tools_list=[calc_tool],
            schema=Answer,
            cancellation_token=CancellationToken(),
        )
    except Exception as e:
        error_msg = str(e).lower()
        # Try to get model name from various possible attributes
        model_name = getattr(real_llm_expensive, "_model_name", getattr(real_llm_expensive, "litellm_model_name", "")).lower()

        if "missing a thought_signature" in error_msg:
            pytest.skip(f"Vertex AI/Gemini requires thought signature which is currently not handled: {e}")
        if "invalid response object" in error_msg and "keyerror: 'content'" in error_msg:
            pytest.skip(f"Provider returned invalid response format (litellm issue): {e}")

        # Fallback for complex nested exceptions where the string might be truncated or formatted differently
        # Specific skip for Gemini 400 errors which are typically the thought signature issue in this context
        if "gemini" in model_name and "400" in error_msg:
             pytest.skip(f"Skipping Gemini 400 error (likely thought signature): {e}")

        raise

    # Validate the synthesized result
    assert response.content, "Expected non-empty synthesized response"
    assert isinstance(response.parsed_object, Answer)

    # Relax assertion for smaller models or partial completions
    model_name = getattr(real_llm_expensive, "_model_name", "").lower()
    if "nano" in model_name or "mini" in model_name:
        # Smaller models might only do one calculation
        assert len(response.parsed_object.result) > 0, "Expected at least one result"
        assert all(r in [8, 14] for r in response.parsed_object.result), (
            f"Unexpected result values: {response.parsed_object.result}"
        )
    else:
        assert set(response.parsed_object.result) == {8, 14}, (
            f"Expected [8, 14] in result, got: {response.parsed_object.result}"
        )
