#!/usr/bin/env python3
"""
Example showing enhanced tool descriptions that guide LLM usage.
"""

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, AgentOutput


class DataAnalyzer(Agent):
    """Agent with enhanced tool description."""
    
    def _get_tool_description(self) -> str:
        """Custom tool description explaining when to use this agent."""
        return (
            "Use this tool when you need to analyze, process, or extract insights from data. "
            "Ideal for: statistical analysis, data cleaning, pattern detection, trend identification, "
            "data visualization recommendations, and quantitative research tasks. "
            "Input should include the data to analyze and the type of analysis required."
        )
    
    def _get_agent_input_schema(self) -> dict[str, any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The data to analyze (CSV, JSON, or descriptive text)"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["statistical", "trend", "pattern", "summary", "visualization"],
                    "description": "Type of analysis to perform"
                },
                "focus": {
                    "type": "string",
                    "description": "Specific aspect to focus on (optional)"
                }
            },
            "required": ["data", "analysis_type"]
        }
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        inputs = message.inputs or {}
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=f"Analyzed {inputs.get('data', 'data')} using {inputs.get('analysis_type', 'general')} analysis",
            metadata={'analysis_complete': True}
        )


class ContentWriter(Agent):
    """Agent using tool_description parameter."""
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        inputs = message.inputs or {}
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=f"Created {inputs.get('content_type', 'content')} about {inputs.get('topic', 'general topic')}",
            metadata={'content_created': True}
        )


class LegalAdvisor(Agent):
    """Agent with basic description that gets enhanced automatically."""
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        inputs = message.inputs or {}
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=f"Legal analysis of: {inputs.get('prompt', 'legal matter')}",
            metadata={'legal_advice': True}
        )


def demo_tool_descriptions():
    """Demonstrate different approaches to tool descriptions."""
    print("🔧 Enhanced Tool Descriptions Demo")
    print("=" * 40)
    
    # 1. Custom override method
    analyzer_config = AgentConfig(
        agent_id='data_analyzer',
        role='ANALYZER',
        description='Analyzes data and provides insights',
        parameters={'model': 'gpt-4'},
        tools=[]
    )
    analyzer = DataAnalyzer(**analyzer_config.model_dump())
    
    # 2. Via tool_description parameter
    writer_config = AgentConfig(
        agent_id='content_writer',
        role='WRITER', 
        description='Creates written content',
        parameters={
            'model': 'gpt-4',
            'tool_description': (
                "Use this tool when you need to create written content such as articles, "
                "blog posts, documentation, marketing copy, or creative writing. "
                "Specify the topic, target audience, tone, and desired length. "
                "Best for: content creation, copywriting, documentation, and creative tasks."
            )
        },
        tools=[]
    )
    writer = ContentWriter(**writer_config.model_dump())
    
    # 3. Auto-enhanced from agent description
    legal_config = AgentConfig(
        agent_id='legal_advisor',
        role='LEGAL',
        description='Review contracts, assess legal risks, and provide compliance guidance',
        parameters={'model': 'gpt-4'},
        tools=[]
    )
    legal = LegalAdvisor(**legal_config.model_dump())
    
    # 4. Minimal agent (fallback description)
    class BasicAgent(Agent):
        async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
            return AgentOutput(agent_id=self.agent_id, outputs="Basic processing")
    
    basic_config = AgentConfig(
        agent_id='basic_agent',
        role='HELPER',
        description='',  # Empty description
        parameters={'model': 'gpt-4'},
        tools=[]
    )
    basic = BasicAgent(**basic_config.model_dump())
    
    # Show the different description styles
    agents = [
        ("Custom Override", analyzer),
        ("Parameter-based", writer), 
        ("Auto-enhanced", legal),
        ("Fallback", basic)
    ]
    
    for desc_type, agent in agents:
        try:
            tool_def = agent.get_autogen_tool_definition()
            print(f"\n📋 {desc_type} ({agent.role}):")
            print(f"   Name: {tool_def['name']}")
            print(f"   Description: {tool_def['description']}")
            print(f"   Length: {len(tool_def['description'])} chars")
        except Exception as e:
            print(f"\n📋 {desc_type} ({agent.role}): Error - {e}")


if __name__ == "__main__":
    demo_tool_descriptions()