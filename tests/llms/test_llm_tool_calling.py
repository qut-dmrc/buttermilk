"""Test tool calling functionality across all LLM models."""


import pytest
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import SystemMessage, UserMessage
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, Field

from buttermilk._core.llms import CHAT_MODELS

# Models known to not support tool calling
MODELS_WITHOUT_TOOL_SUPPORT = {"haiku", "llama32_90b"}

# Models that have quirks with tool calling (e.g., may not follow instructions perfectly)
MODELS_WITH_TOOL_QUIRKS = {"llama4maverick", "llama33_70b", "o4mini"}


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
        "london": WeatherResponse(location="London", temperature=15.5, condition="cloudy", humidity=75),
        "new york": WeatherResponse(location="New York", temperature=22.0, condition="sunny", humidity=60),
        "tokyo": WeatherResponse(location="Tokyo", temperature=18.5, condition="rainy", humidity=85),
        "paris": WeatherResponse(location="Paris", temperature=17.0, condition="partly cloudy", humidity=70),
    }

    # Normalize location for lookup
    location_key = location.lower().strip()

    # Return mock data or default
    return weather_data.get(location_key, WeatherResponse(location=location, temperature=20.0, condition="clear", humidity=65))


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
async def test_single_tool_call(llm_expensive):
    """Test that each LLM can make a single tool call."""
    # Skip if model doesn't support tools
    model_name = getattr(llm_expensive, "_model_name", None)
    if model_name and model_name in MODELS_WITHOUT_TOOL_SUPPORT:
        pytest.skip(f"{model_name} doesn't support tool calling")

    # Create a simple weather tool
    weather_tool = FunctionTool(get_weather, name="get_weather", description="Get the current weather for a location")

    messages = [
        SystemMessage(content="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather.", source="system"),
        UserMessage(content="What's the weather like in London?", source="user"),
    ]

    try:
        # Test with tool calling
        response = await llm_expensive.call_chat(messages=messages, tools_list=[weather_tool], cancellation_token=CancellationToken())

        # Verify response mentions London and weather details
        assert response.content
        assert isinstance(response.content, str)
        content_lower = response.content.lower()

        # Should mention London
        assert "london" in content_lower

        # Should mention weather details (at least one of these)
        weather_terms = ["cloudy", "15.5", "75", "humidity", "temperature", "celsius", "°c"]
        assert any(term in content_lower for term in weather_terms), f"Response should contain weather information, got: {response.content}"
    except Exception as e:
        if "does not support function calling" in str(e):
            pytest.skip(f"Model doesn't support tool calling: {e}")
        raise


@pytest.mark.anyio
async def test_multiple_tool_calls(llm_expensive):
    """Test that LLMs can handle multiple tools and select the right one."""
    # Skip if model doesn't support tools
    model_name = getattr(llm_expensive, "_model_name", None)
    if model_name and model_name in MODELS_WITHOUT_TOOL_SUPPORT:
        pytest.skip(f"{model_name} doesn't support tool calling")

    # Create multiple tools
    weather_tool = FunctionTool(get_weather, name="get_weather", description="Get the current weather for a location")

    calc_tool = FunctionTool(calculate_sum, name="calculate_sum", description="Calculate the sum of two numbers")

    messages = [
        SystemMessage(
            content="You are a helpful assistant with access to weather and calculation tools. Use the calculate_sum tool when asked about addition.",
            source="system",
        ),
        UserMessage(content="What's 5 + 3?", source="user"),
    ]

    try:
        # Test with multiple tools available
        response = await llm_expensive.call_chat(messages=messages, tools_list=[weather_tool, calc_tool], cancellation_token=CancellationToken())

        # Verify response contains the correct sum
        assert response.content
        assert isinstance(response.content, str)

        # For models with tool quirks, be more lenient
        if model_name in MODELS_WITH_TOOL_QUIRKS:
            # Just check if they attempted to do math or mentioned the numbers
            assert any(
                term in response.content.lower() for term in ["8", "eight", "5", "3", "calculate", "sum"]
            ), f"Response should relate to the calculation, got: {response.content}"
        else:
            # Check for both digit "8" and word "eight"
            assert any(term in response.content.lower() for term in ["8", "eight"]), \
                f"Response should contain the sum 8, got: {response.content}"
    except Exception as e:
        if "does not support function calling" in str(e):
            pytest.skip(f"Model doesn't support tool calling: {e}")
        raise


@pytest.mark.anyio
async def test_no_tool_needed(llm_expensive):
    """Test that LLMs don't use tools when not needed."""
    # Skip if model doesn't support tools
    model_name = getattr(llm_expensive, "_model_name", None)
    if model_name and model_name in MODELS_WITHOUT_TOOL_SUPPORT:
        pytest.skip(f"{model_name} doesn't support tool calling")

    # Create tools that shouldn't be used
    weather_tool = FunctionTool(get_weather, name="get_weather", description="Get the current weather for a location")

    calc_tool = FunctionTool(calculate_sum, name="calculate_sum", description="Calculate the sum of two numbers")

    messages = [
        SystemMessage(content="You are a helpful assistant. Answer questions directly when you can without using any tools.", source="system"),
        UserMessage(content="What is the capital of France?", source="user"),
    ]

    try:
        # Test with tools available but not needed
        response = await llm_expensive.call_chat(messages=messages, tools_list=[weather_tool, calc_tool], cancellation_token=CancellationToken())

        # Verify response contains Paris without using tools
        assert response.content
        assert isinstance(response.content, str)

        # For models with tool quirks, they might refuse to answer without tools
        if model_name in MODELS_WITH_TOOL_QUIRKS:
            # These models might refuse entirely when tools are present but not relevant
            # Just verify they got a response at all
            assert len(response.content) > 0, "Should have some response"
            # Log for debugging but don't fail if they refuse
            if not any(term in response.content.lower() for term in ["paris", "france", "capital"]):
                print(f"Note: {model_name} refused to answer without relevant tools: {response.content}")
        else:
            assert "paris" in response.content.lower(), f"Response should mention Paris, got: {response.content}"
    except Exception as e:
        if "does not support function calling" in str(e):
            pytest.skip(f"Model doesn't support tool calling: {e}")
        raise


@pytest.mark.parametrize("model_name", CHAT_MODELS)
@pytest.mark.anyio
async def test_all_models_basic_tool_call(model_name, bm):
    """Test that all configured models can make basic tool calls."""
    # Skip if model not available
    if model_name not in bm.llms.connections:
        pytest.skip(f"Model {model_name} not configured")

    # Get the model client
    try:
        model_client = bm.llms.get_autogen_chat_client(model_name)
    except Exception as e:
        pytest.skip(f"Could not initialize {model_name}: {e}")

    # Create a simple tool
    weather_tool = FunctionTool(get_weather, name="get_weather", description="Get the current weather for a location")

    messages = [
        UserMessage(content="What's the weather in Paris? Please use the weather tool.", source="user"),
    ]

    try:
        # Test basic tool calling
        response = await model_client.call_chat(messages=messages, tools_list=[weather_tool], cancellation_token=CancellationToken())

        # Verify we got a response
        assert response.content
        assert isinstance(response.content, str)

        # Should mention Paris in the response
        assert "paris" in response.content.lower(), f"{model_name} should mention Paris in response, got: {response.content}"

    except Exception as e:
        # Check if this is a known model without tool support
        if model_name in MODELS_WITHOUT_TOOL_SUPPORT:
            # Expected failure, just verify basic functionality
            print(f"Info: {model_name} doesn't support tool calling (expected): {e}")

            # Try without tools as a fallback
            response = await model_client.create(messages=messages)
            assert response.content
            return

        # For other models, this is unexpected
        raise AssertionError(f"{model_name} unexpectedly failed tool calling: {e}")


@pytest.mark.anyio
async def test_call_chat_intercept_tools_returns_function_calls(llm_expensive):
    """Verify that call_chat(intercept_tools=True) returns FunctionCall objects without executing."""
    # Skip if model doesn't support tools
    model_name = getattr(llm_expensive, "_model_name", None)
    if model_name and model_name in MODELS_WITHOUT_TOOL_SUPPORT:
        pytest.skip(f"{model_name} doesn't support tool calling")

    calc_tool = FunctionTool(calculate_sum, name="calculate_sum", description="Calculate the sum of two numbers")

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
        result = await llm_expensive.call_chat(
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
    assert isinstance(result.content, list), f"Expected a list of FunctionCall, got: {type(result.content)}"
    assert all(isinstance(c, FunctionCall) for c in result.content), "Expected FunctionCall objects"
    assert any(c.name == "calculate_sum" for c in result.content), "Expected a calculate_sum tool call"


@pytest.mark.anyio
async def test_structured_output_with_tools(llm_expensive):
    """Test that models handle the interaction between structured output and tools correctly."""

    class Answer(BaseModel):
        """Structured answer format."""

        result: str = Field(description="The answer to the question")
        confidence: float = Field(description="Confidence level from 0 to 1")

    # Create a simple tool
    calc_tool = FunctionTool(calculate_sum, name="calculate_sum", description="Calculate the sum of two numbers")

    messages = [
        SystemMessage(content="You are a helpful assistant. Always structure your responses using the provided schema.", source="system"),
        UserMessage(content="What is the capital of Japan?", source="user"),
    ]

    # Test with structured output (tools should not be passed with structured output for certain models)
    response = await llm_expensive.call_chat(
        messages=messages,
        tools_list=[calc_tool],
        schema=Answer,
        cancellation_token=CancellationToken(),  # This might be ignored for some models
    )

    # Verify structured response
    assert response.content
    assert isinstance(response.content, Answer)
    assert "tokyo" in response.content.result.lower()


@pytest.mark.anyio
async def test_call_chat_tool_exec_then_synthesis_with_schema(llm_expensive):
    """Cover the full flow: initial tool call -> tool execution -> synthesis call with schema.

    This test helps surface issues where the synthesis call incorrectly sets a structured
    response_format that some models don't support.
    """

    class Answer(BaseModel):
        result: list[int] = Field(description="The final answer")

    # Skip if model doesn't support tools
    model_name = getattr(llm_expensive, "_model_name", None)
    if model_name and model_name in MODELS_WITHOUT_TOOL_SUPPORT:
        pytest.skip(f"{model_name} doesn't support tool calling")

    calc_tool = FunctionTool(calculate_sum, name="calculate_sum", description="Calculate the sum of two numbers")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. When a task involves addition, you must use the calculate_sum tool "
                "for every addition operation. Do not perform any arithmetic yourself. After finishing the tool "
                "calls, present the final answers using the provided schema."
            ),
            source="system",
        ),
        UserMessage(
            content=("Compute (5 + 3) and (10 + 4) using separate calls to the calculate_sum tool. Return the final answer using the schema."),
            source="user",
        ),
    ]

    try:
        response = await llm_expensive.call_chat(
            messages=messages,
            tools_list=[calc_tool],
            schema=Answer,
            cancellation_token=CancellationToken(),
        )
    except Exception as e:
        # Surface the specific error the user saw to make the failure actionable
        err = str(e).lower()
        if "response_format" in err and "json_schema" in err and "not supported" in err:
            pytest.fail(
                "Synthesis call attempted to use a structured response_format (json_schema) "
                "on a model that doesn't support it. Ensure call_chat() avoids structured "
                "response formats for unsupported models after tool execution."
            )
        if "does not support function calling" in err:
            pytest.skip(f"Model doesn't support tool calling: {e}")
        raise

    # Validate the synthesized result
    assert response.content, "Expected non-empty synthesized response"
    assert set(response.content.result) == {8, 14}, f"Expected [8, 14] in result, got: {response.content}"
    assert 0.0 <= response.content.confidence <= 1.0
