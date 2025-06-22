#!/usr/bin/env python3
"""
Example showing agents with multiple tools.
"""

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.mcp_decorators import tool


class MultiToolResearcher(Agent):
    """Researcher agent with multiple specialized tools."""
    
    def _get_agent_input_schema(self) -> dict[str, any]:
        """Main tool schema for general research."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "Research query or question"
                },
                "depth": {
                    "type": "string",
                    "enum": ["shallow", "medium", "deep"],
                    "description": "Research depth"
                }
            },
            "required": ["query"]
        }
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        """Main research process."""
        inputs = message.inputs or {}
        query = inputs.get('query', '')
        depth = inputs.get('depth', 'medium')
        
        result = f"Researched '{query}' with {depth} depth"
        
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=result,
            metadata={'query': query, 'depth': depth}
        )
    
    @tool(name="fact_check", description="Verify facts and claims")
    async def fact_check(self, claim: str, sources: list[str] = None) -> dict:
        """Check the accuracy of a claim."""
        return {
            'claim': claim,
            'verified': True,
            'confidence': 0.95,
            'sources_checked': len(sources or [])
        }
    
    @tool(name="find_sources", description="Find reliable sources for a topic")
    async def find_sources(self, topic: str, source_type: str = "academic") -> dict:
        """Find sources for research."""
        return {
            'topic': topic,
            'source_type': source_type,
            'sources_found': [
                f"Source 1 about {topic}",
                f"Source 2 about {topic}",
                f"Source 3 about {topic}"
            ]
        }
    
    @tool(name="summarize_research", description="Summarize research findings")
    async def summarize_research(self, findings: list[str], max_length: int = 500) -> dict:
        """Summarize research findings."""
        return {
            'summary': f"Summary of {len(findings)} findings (max {max_length} chars)",
            'key_points': findings[:3],  # Top 3 points
            'total_findings': len(findings)
        }


if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🔬 Multi-Tool Agent Demo")
        print("=" * 30)
        
        # Create the agent
        config = AgentConfig(
            agent_id='multi_researcher',
            role='RESEARCHER',
            description='Advanced researcher with specialized tools',
            parameters={'model': 'gpt-4'},
            tools=[]
        )
        agent = MultiToolResearcher(**config.model_dump())
        
        # Show main tool (automatic)
        main_tool = agent.get_autogen_tool_definition()
        print("🔧 Main Tool (Automatic):")
        print(f"  Name: {main_tool['name']}")
        print(f"  Description: {main_tool['description']}")
        print(f"  Schema: {list(main_tool['input_schema']['properties'].keys())}")
        
        # Show additional tools (from @tool decorators)
        additional_tools = agent.get_tool_definitions()
        print(f"\n🛠️  Additional Tools ({len(additional_tools)}):")
        for tool_def in additional_tools:
            print(f"  - {tool_def.name}: {tool_def.description}")
        
        # Show announcement includes both
        announcement = agent.create_announcement('initial', 'joining')
        print(f"\n📢 Announcement:")
        print(f"  Main tool: {announcement.tool_definition['name']}")
        print(f"  Available tools: {announcement.available_tools}")
        
        # Test the tools
        print("\n🧪 Testing Main Tool:")
        main_input = AgentInput(inputs={'query': 'AI ethics', 'depth': 'deep'})
        main_result = await agent._process(message=main_input)
        print(f"  Result: {main_result.outputs}")
        
        print("\n🧪 Testing Additional Tools:")
        fact_result = await agent.fact_check("AI is beneficial", ["source1", "source2"])
        print(f"  Fact check: {fact_result}")
        
        sources_result = await agent.find_sources("machine learning", "academic")
        print(f"  Sources: {len(sources_result['sources_found'])} found")
    
    asyncio.run(demo())