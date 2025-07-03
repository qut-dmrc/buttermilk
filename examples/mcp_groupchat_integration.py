"""Example showing MCP adapter integration with StructuredLLMHostAgent.

This example demonstrates how the MCP adapter works alongside the existing
groupchat infrastructure, allowing the same agents to be used in both contexts.
"""

import asyncio
from typing import Any
from unittest.mock import Mock

from buttermilk._core import Agent, AgentInput, AgentOutput
from buttermilk._core.mcp_decorators import tool
from buttermilk._core.tool_definition import UnifiedRequest
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk.mcp.autogen_adapter import MCPHostProvider
from buttermilk.mcp.tool_registry import get_tool_discovery_service


class TextProcessorAgent(Agent):
    """Text processing agent with multiple tools."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process text manipulation requests."""
        operation = message.inputs.get("operation", "uppercase")
        text = message.inputs.get("text", "")
        
        if operation == "uppercase":
            result = self.to_uppercase(text)
        elif operation == "word_count":
            result = self.count_words(text)
        elif operation == "reverse":
            result = self.reverse_text(text)
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs={"result": result}
        )
    
    @tool
    def to_uppercase(self, text: str) -> str:
        """Convert text to uppercase.
        
        Args:
            text: Text to convert
            
        Returns:
            Uppercase text
        """
        return text.upper()
    
    @tool
    def count_words(self, text: str) -> dict[str, Any]:
        """Count words in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Word count statistics
        """
        words = text.split()
        unique_words = set(word.lower().strip('.,!?;:"()[]') for word in words)
        
        return {
            "total_words": len(words),
            "unique_words": len(unique_words),
            "characters": len(text),
            "characters_no_spaces": len(text.replace(" ", ""))
        }
    
    @tool
    def reverse_text(self, text: str) -> str:
        """Reverse the given text.
        
        Args:
            text: Text to reverse
            
        Returns:
            Reversed text
        """
        return text[::-1]


class SentimentAgent(Agent):
    """Simple sentiment analysis agent."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process sentiment analysis requests."""
        text = message.inputs.get("text", "")
        result = await self.analyze_sentiment(text)
        
        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs={"result": result}
        )
    
    @tool
    async def analyze_sentiment(self, text: str) -> dict[str, Any]:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        # Simple rule-based sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_words_found": positive_count,
            "negative_words_found": negative_count,
            "text_length": len(text)
        }


async def demo_structured_host_with_tools():
    """Demonstrate StructuredLLMHostAgent using tools from registered agents."""
    print("\n=== StructuredLLMHostAgent with Tool Discovery ===")
    
    # Create agent instances
    text_agent = TextProcessorAgent(
        agent_name="text_processor",
        role="processor",
        session_id="host-demo"
    )
    
    sentiment_agent = SentimentAgent(
        agent_name="sentiment_analyzer",
        role="analyzer", 
        session_id="host-demo"
    )
    
    # Register agents with discovery service
    discovery_service = get_tool_discovery_service()
    discovery_service.register_agent(text_agent)
    discovery_service.register_agent(sentiment_agent)
    
    # Show discovered tools
    all_tools = discovery_service.get_all_tools()
    print(f"Discovered {len(all_tools)} tools from agents:")
    for tool in all_tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Create mock structured host (would normally use real StructuredLLMHostAgent)
    print("\nSimulating StructuredLLMHostAgent tool usage:")
    
    # The host would discover tools like this:
    available_tools = [tool["tool_definition"] for tool in all_tools]
    print(f"Host discovered {len(available_tools)} tool definitions")
    
    # Simulate host deciding to call text processing tool
    print("\n1. Host decides to call text processing tool...")
    unified_request = UnifiedRequest.from_groupchat_step(
        agent_role="processor",
        inputs={"text": "Hello World! This is a great example."},
        tool_name="count_words"
    )
    
    result = await text_agent.handle_unified_request(unified_request)
    print(f"Word count result: {result}")
    
    # Simulate host deciding to call sentiment analysis
    print("\n2. Host decides to call sentiment analysis tool...")
    unified_request = UnifiedRequest.from_groupchat_step(
        agent_role="analyzer",
        inputs={"text": "This is an amazing and wonderful example!"}
    )
    
    result = await sentiment_agent.handle_unified_request(unified_request)
    print(f"Sentiment analysis result: {result}")


async def demo_mcp_exposure():
    """Demonstrate exposing the same agents via MCP."""
    print("\n=== MCP Exposure of Same Agents ===")
    
    # Get agents from discovery service
    discovery_service = get_tool_discovery_service()
    all_tools = discovery_service.get_all_tools()
    
    if not all_tools:
        print("No tools found - run structured host demo first")
        return
    
    # Initialize MCP provider
    mcp_provider = MCPHostProvider(port=8789)
    
    # The agents are already registered in the discovery service,
    # so the MCP provider can access them
    available_mcp_tools = mcp_provider.list_available_tools()
    print(f"MCP exposed {len(available_mcp_tools)} tools:")
    for tool_key, tool_info in available_mcp_tools.items():
        print(f"  - {tool_key}: {tool_info['description']}")
    
    # Test MCP tool invocation
    print("\nTesting MCP tool invocation:")
    
    try:
        # Call text processing via MCP
        result = await mcp_provider.invoke_tool(
            "text_processor_processor.to_uppercase",
            {"text": "hello from mcp!"}
        )
        print(f"MCP uppercase result: {result}")
        
        # Call sentiment analysis via MCP
        result = await mcp_provider.invoke_tool(
            "sentiment_analyzer_analyzer.analyze_sentiment",
            {"text": "This MCP integration is fantastic and excellent!"}
        )
        print(f"MCP sentiment result: {result}")
        
    except Exception as e:
        print(f"MCP error: {e}")


async def demo_unified_request_contexts():
    """Demonstrate how unified requests work in different contexts."""
    print("\n=== Unified Request Context Demo ===")
    
    # Create agent
    text_agent = TextProcessorAgent(
        agent_name="context_demo",
        role="processor",
        session_id="context-test"
    )
    
    # Test MCP context
    print("1. MCP Context Request:")
    mcp_request = UnifiedRequest.from_mcp_call(
        tool_name="reverse_text",
        parameters={"text": "MCP Context"},
        agent_name="context_demo_processor"
    )
    print(f"   Is MCP: {mcp_request.is_mcp_request}")
    print(f"   Is Groupchat: {mcp_request.is_groupchat_request}")
    print(f"   Metadata: {mcp_request.metadata}")
    
    result = await text_agent.handle_unified_request(mcp_request)
    print(f"   Result: {result}")
    
    # Test groupchat context
    print("\n2. Groupchat Context Request:")
    groupchat_request = UnifiedRequest.from_groupchat_step(
        agent_role="processor",
        inputs={"text": "Groupchat Context"},
        tool_name="reverse_text"
    )
    print(f"   Is MCP: {groupchat_request.is_mcp_request}")
    print(f"   Is Groupchat: {groupchat_request.is_groupchat_request}")
    print(f"   Metadata: {groupchat_request.metadata}")
    
    result = await text_agent.handle_unified_request(groupchat_request)
    print(f"   Result: {result}")


async def main():
    """Main demonstration function."""
    print("🔄 Buttermilk MCP-Groupchat Integration Demo")
    print("=============================================")
    
    # Demo 1: Show how StructuredLLMHostAgent can discover and use tools
    await demo_structured_host_with_tools()
    
    # Demo 2: Show how the same agents can be exposed via MCP
    await demo_mcp_exposure()
    
    # Demo 3: Show unified request handling in different contexts
    await demo_unified_request_contexts()
    
    print("\n✅ Integration demo completed!")
    print("\nKey integration points:")
    print("- Same agents work in both groupchat and MCP contexts")
    print("- Tool discovery service provides unified agent registry")
    print("- UnifiedRequest handles both contexts seamlessly")
    print("- StructuredLLMHostAgent can use dynamically discovered tools")
    print("- MCP adapter exposes tools without code changes")


if __name__ == "__main__":
    asyncio.run(main())