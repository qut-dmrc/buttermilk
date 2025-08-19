"""Test structured output workaround for models without native structured output support."""

import pytest
from autogen_core import CancellationToken
from autogen_core.models import SystemMessage, UserMessage, ModelInfo, ModelFamily, RequestUsage
from pydantic import BaseModel, Field

from buttermilk._core.llms import AutoGenWrapper, ModelOutput


class PersonInfo(BaseModel):
    """Schema for person information."""
    
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")


class BookRecommendation(BaseModel):
    """Schema for book recommendations."""
    
    title: str = Field(description="The book title")
    author: str = Field(description="The book author")
    genre: str = Field(description="The book genre")
    reason: str = Field(description="Why this book is recommended")


@pytest.mark.anyio
async def test_structured_output_fake_tool_workaround():
    """Test that models without structured output use fake tool workaround."""
    # Create a mock client that doesn't support structured output
    from unittest.mock import AsyncMock, MagicMock
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult
    
    mock_client = AsyncMock()
    
    # Model info indicating no structured output support but has function calling
    model_info = ModelInfo(
        family=ModelFamily.UNKNOWN,
        structured_output=False,
        function_calling=True,
        vision=False,
        json_output=False,
    )
    
    # Create wrapper - Use pydantic's validation_alias to bypass validation
    wrapper = AutoGenWrapper.model_construct(client=mock_client, model_info=model_info)
    
    # Mock the client to return a tool call
    mock_response = CreateResult(
        content=[FunctionCall(
            id="call_123",
            name="create_personinfo",
            arguments='{"name": "John Doe", "age": 30, "city": "New York"}'
        )],
        finish_reason="stop",
        usage=RequestUsage(prompt_tokens=10, completion_tokens=20),
        cached=False
    )
    mock_client.create.return_value = mock_response
    
    # Test calling with schema
    messages = [
        SystemMessage(content="Extract person information.", source="system"),
        UserMessage(content="John Doe is 30 years old and lives in New York.", source="user")
    ]
    
    result = await wrapper.create(
        messages=messages,
        schema=PersonInfo,
        cancellation_token=CancellationToken()
    )
    
    # Verify the fake tool was created and used
    assert mock_client.create.called
    call_args = mock_client.create.call_args
    
    # Check that tools were passed (our fake schema tool)
    assert "tools" in call_args.kwargs
    assert len(call_args.kwargs["tools"]) == 1
    assert call_args.kwargs["tools"][0].name == "create_personinfo"
    
    # Verify structured output was not requested
    assert call_args.kwargs.get("json_output") is False
    
    # Verify the result
    assert isinstance(result, ModelOutput)
    assert result.content is not None
    assert isinstance(result.content, PersonInfo)
    assert result.content.name == "John Doe"
    assert result.content.age == 30
    assert result.content.city == "New York"


@pytest.mark.anyio
async def test_structured_output_native_support():
    """Test that models with native structured output don't use fake tool."""
    from unittest.mock import AsyncMock, MagicMock
    from autogen_core.models import CreateResult
    
    mock_client = AsyncMock()
    
    # Model info indicating structured output support
    model_info = ModelInfo(
        family=ModelFamily.UNKNOWN,
        structured_output=True,
        function_calling=True,
        vision=False,
        json_output=True,
    )
    
    # Create wrapper - Use pydantic's validation_alias to bypass validation
    wrapper = AutoGenWrapper.model_construct(client=mock_client, model_info=model_info)
    
    # Mock the client to return structured output directly
    mock_response = CreateResult(
        content='{"title": "Nineteen Eighty-Four", "author": "George Orwell", "genre": "Dystopian", "reason": "A classic"}',
        finish_reason="stop",
        usage=RequestUsage(prompt_tokens=10, completion_tokens=20),
        cached=False
    )
    mock_client.create.return_value = mock_response
    
    # Test calling with schema
    messages = [
        UserMessage(content="Recommend a classic book.", source="user")
    ]
    
    result = await wrapper.create(
        messages=messages,
        schema=BookRecommendation,
        cancellation_token=CancellationToken()
    )
    
    # Verify native structured output was used
    assert mock_client.create.called
    call_args = mock_client.create.call_args
    
    # Check that json_output was requested with the schema
    assert call_args.kwargs.get("json_output") == BookRecommendation
    
    # Check that no tools were passed
    assert "tools" not in call_args.kwargs or len(call_args.kwargs.get("tools", [])) == 0
    
    # Verify the result is parsed correctly
    assert isinstance(result, ModelOutput)
    assert result.content is not None
    assert isinstance(result.content, BookRecommendation)
    assert result.content.title == "Nineteen Eighty-Four"


@pytest.mark.anyio
async def test_no_fake_tool_when_tools_provided():
    """Test that fake tool is not created when actual tools are provided."""
    from unittest.mock import AsyncMock, MagicMock
    from autogen_core.tools import FunctionTool
    from autogen_core.models import CreateResult
    
    mock_client = AsyncMock()
    
    # Model info indicating no structured output support
    model_info = ModelInfo(
        family=ModelFamily.UNKNOWN,
        structured_output=False,
        function_calling=True,
        vision=False,
        json_output=False,
    )
    
    # Create wrapper - Use pydantic's validation_alias to bypass validation
    wrapper = AutoGenWrapper.model_construct(client=mock_client, model_info=model_info)
    
    # Create a real tool
    async def get_time() -> str:
        """Get current time."""
        return "12:00 PM"
    
    time_tool = FunctionTool(
        func=get_time,
        name="get_time",
        description="Get the current time"
    )
    
    # Mock response
    mock_response = CreateResult(
        content="The current time is 12:00 PM",
        finish_reason="stop",
        usage=RequestUsage(prompt_tokens=10, completion_tokens=20),
        cached=False
    )
    mock_client.create.return_value = mock_response
    
    # Test calling with schema AND tools
    messages = [
        UserMessage(content="What time is it?", source="user")
    ]
    
    # When tools are provided, the fake schema tool should NOT be created
    result = await wrapper.create(
        messages=messages,
        tools=[time_tool],
        schema=PersonInfo,  # This should be ignored when tools are provided
        cancellation_token=CancellationToken()
    )
    
    # Verify only the provided tool was used, no fake tool
    assert mock_client.create.called
    call_args = mock_client.create.call_args
    
    # Check that only the real tool was passed
    assert "tools" in call_args.kwargs
    assert len(call_args.kwargs["tools"]) == 1
    assert call_args.kwargs["tools"][0].name == "get_time"
    
    # Verify structured output was not requested
    assert call_args.kwargs.get("json_output") is False