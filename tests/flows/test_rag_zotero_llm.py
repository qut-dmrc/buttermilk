"""Test RAG Zotero agent with actual LLM calls and structured outputs."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, Field

from buttermilk._core import AgentInput
from buttermilk.agents.rag.rag_zotero import RagZotero, ZoteroResearchResult, ZoteroReference
from buttermilk._core.llms import CHATMODELS


# Mock search results to simulate vector search
MOCK_SEARCH_RESULTS = [
    {
        "text": "Social media usage has been linked to increased anxiety in teenagers according to recent studies.",
        "metadata": {
            "source": "Smith et al., 2023", 
            "citation": "Smith, J., Brown, A., & Davis, C. (2023). The impact of social media on teenage mental health. Journal of Psychology, 45(3), 234-251.",
            "doi": "10.1037/jap.2023.045"
        }
    },
    {
        "text": "However, moderate social media use can also provide social support and connection for isolated teens.",
        "metadata": {
            "source": "Johnson & Lee, 2023",
            "citation": "Johnson, M., & Lee, K. (2023). Positive aspects of social media for adolescents. Cyberpsychology Review, 12(2), 89-102.", 
            "doi": "10.1089/cyber.2023.012"
        }
    }
]


async def mock_vector_search(query: str, k: int = 5) -> list[dict]:
    """Mock vector search function that returns fake Zotero results."""
    return MOCK_SEARCH_RESULTS[:k]


@pytest.mark.parametrize("model_name", CHATMODELS)
@pytest.mark.anyio
async def test_rag_zotero_with_structured_output(model_name, bm):
    """Test RagZotero agent with each LLM model for structured output generation."""
    
    # Skip if model not configured
    if model_name not in bm.llms.connections:
        pytest.skip(f"Model {model_name} not configured")
    
    # Create the agent with the specific model
    # Include tools to simulate the actual flow configuration
    agent = RagZotero(
        agent_id="test_zotero",
        agent_name="Test Zotero",
        role="ZOTERO_RESEARCHER",
        parameters={
            "model": model_name,
            "template": "rag"
        },
        tools={
            "vector_search": FunctionTool(
                mock_vector_search,
                name="vector_search",
                description="Search the Zotero vector database for relevant academic papers"
            )
        }
    )
    
    # Initialize the agent
    await agent.initialize()
    
    # Create input
    agent_input = AgentInput(
        inputs={
            "prompt": "What is the impact of social media on teenage mental health?",
            "context": ""
        }
    )
    
    try:
        # Process the request
        result = await agent(agent_input)
        
        # Verify the output structure
        assert hasattr(result, 'outputs'), f"{model_name}: Result should have outputs attribute"
        
        # Check if it's the expected structured output
        if isinstance(result.outputs, ZoteroResearchResult):
            # Perfect - got structured output directly
            research_result = result.outputs
        elif isinstance(result.outputs, dict) and 'literature' in result.outputs:
            # Try to parse from dict
            research_result = ZoteroResearchResult(**result.outputs)
        elif isinstance(result.outputs, str):
            # Some models might return JSON string
            import json
            try:
                data = json.loads(result.outputs)
                research_result = ZoteroResearchResult(**data)
            except:
                pytest.fail(f"{model_name}: Could not parse structured output from string: {result.outputs[:200]}")
        else:
            pytest.fail(f"{model_name}: Unexpected output type: {type(result.outputs)}")
        
        # Validate the research result
        assert isinstance(research_result.literature, list), f"{model_name}: literature should be a list"
        assert len(research_result.literature) > 0, f"{model_name}: Should have at least one reference"
        
        # Check first reference
        first_ref = research_result.literature[0]
        assert isinstance(first_ref, ZoteroReference), f"{model_name}: References should be ZoteroReference objects"
        assert first_ref.summary, f"{model_name}: Reference should have a summary"
        assert first_ref.source, f"{model_name}: Reference should have a source"
        assert first_ref.citation, f"{model_name}: Reference should have a citation"
        
        # Check response
        assert research_result.response, f"{model_name}: Should have a synthesized response"
        assert len(research_result.response) > 50, f"{model_name}: Response should be substantive"
        
        print(f"✅ {model_name}: Successfully generated structured Zotero output")
        print(f"   - References: {len(research_result.literature)}")
        print(f"   - Response length: {len(research_result.response)}")
        
    except Exception as e:
        # Check if this is the specific error mentioned
        if "Error code: 500" in str(e) and "Internal error encountered" in str(e):
            pytest.fail(f"{model_name}: Got internal server error (500) - this is the issue to fix: {e}")
        else:
            # Log other errors for debugging
            print(f"❌ {model_name}: Error during RAG Zotero test: {e}")
            raise


@pytest.mark.anyio
async def test_rag_zotero_llama4_specific(bm):
    """Specific test for llama4maverick to debug the 500 error."""
    
    model_name = "llama4maverick"
    
    if model_name not in bm.llms.connections:
        pytest.skip(f"Model {model_name} not configured")
    
    # Get the LLM client directly
    llm_client = bm.llms.get_autogen_chat_client(model_name)
    
    # First test: Simple tool calling
    print(f"\n1. Testing simple tool calling with {model_name}...")
    
    async def echo_func(x: str) -> str:
        """Echo the input string."""
        return f"Echo: {x}"
    
    simple_tool = FunctionTool(
        echo_func,
        name="echo",
        description="Echo the input"
    )
    
    try:
        from autogen_core.models import UserMessage
        from autogen_core import CancellationToken
        
        response = await llm_client.call_chat(
            messages=[UserMessage(content="Test echo hello", source="user")],
            tools_list=[simple_tool],
            cancellation_token=CancellationToken()
        )
        print(f"✅ Simple tool calling works: {response.content[:100]}")
    except Exception as e:
        print(f"❌ Simple tool calling failed: {e}")
        if "Error code: 500" in str(e):
            pytest.fail(f"Got 500 error in simple tool calling: {e}")
    
    # Second test: Structured output without tools
    print(f"\n2. Testing structured output without tools...")
    
    class SimpleOutput(BaseModel):
        message: str
        confidence: float
    
    try:
        response = await llm_client.call_chat(
            messages=[UserMessage(content="Say hello with confidence 0.9", source="user")],
            schema=SimpleOutput,
            cancellation_token=CancellationToken()
        )
        print(f"✅ Structured output works: {response.content[:100]}")
    except Exception as e:
        print(f"❌ Structured output failed: {e}")
    
    # Third test: Complex structured output (like ZoteroResearchResult)
    print(f"\n3. Testing complex structured output...")
    
    try:
        response = await llm_client.call_chat(
            messages=[
                UserMessage(
                    content="Generate a research result with one reference about AI",
                    source="user"
                )
            ],
            schema=ZoteroResearchResult,
            cancellation_token=CancellationToken()
        )
        print(f"✅ Complex structured output works: {response.content[:100]}")
    except Exception as e:
        print(f"❌ Complex structured output failed: {e}")
        
        # Try with simpler schema
        print("\n4. Testing with simplified schema...")
        
        class SimpleReference(BaseModel):
            summary: str
            source: str
        
        class SimpleResult(BaseModel):
            literature: list[SimpleReference]
            response: str
        
        try:
            response = await llm_client.call_chat(
                messages=[
                    UserMessage(
                        content="Generate a research result with one reference about AI", 
                        source="user"
                    )
                ],
                schema=SimpleResult,
                cancellation_token=CancellationToken()
            )
            print(f"✅ Simplified structured output works: {response.content[:100]}")
        except Exception as e2:
            print(f"❌ Even simplified structured output failed: {e2}")
    
    # Fifth test: Structured output WITH tools (the problematic combination)
    print(f"\n5. Testing structured output WITH tools...")
    
    try:
        response = await llm_client.call_chat(
            messages=[UserMessage(content="Test echo hello and return result in structured format", source="user")],
            tools_list=[simple_tool],
            schema=SimpleOutput,
            cancellation_token=CancellationToken()
        )
        print(f"✅ Structured output WITH tools works: {response.content[:100]}")
    except Exception as e:
        print(f"❌ Structured output WITH tools failed: {e}")
        if "Error code: 500" in str(e):
            print("   This is the core issue - llama4maverick fails with tools + structured output!")
    
    # Sixth test: Complex structured output WITH tools (like the real agent)
    print(f"\n6. Testing complex structured output WITH tools (like real RagAgent)...")
    
    try:
        response = await llm_client.call_chat(
            messages=[
                UserMessage(
                    content="Search for information and return a research result",
                    source="user"
                )
            ],
            tools_list=[simple_tool],  # Using simple tool to isolate the issue
            schema=ZoteroResearchResult,
            cancellation_token=CancellationToken()
        )
        print(f"✅ Complex structured output WITH tools works!")
    except Exception as e:
        print(f"❌ Complex structured output WITH tools failed: {e}")
        if "Error code: 500" in str(e):
            print("   Confirmed: llama4maverick cannot handle tools + complex structured output together")


if __name__ == "__main__":
    # Allow running specific tests manually
    import asyncio
    from buttermilk._core.dmrc import get_bm, set_bm
    from buttermilk._core.bm_init import BM
    
    async def main():
        # Initialize BM singleton
        bm = BM(name="buttermilk", job="testing")
        await bm.setup()
        set_bm(bm)
        
        # Run test
        await test_rag_zotero_llama4_specific(bm)
    
    asyncio.run(main())