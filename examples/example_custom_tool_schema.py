#!/usr/bin/env python3
"""
Example showing how to customize agent tool schemas.
"""

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, AgentOutput


class AnalyzerAgent(Agent):
    """Agent with custom tool schema for structured analysis."""
    
    def _get_agent_input_schema(self) -> dict[str, any]:
        """Custom input schema for this agent."""
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["sentiment", "topic", "summary"],
                    "description": "Type of analysis to perform"
                },
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum confidence for results"
                }
            },
            "required": ["data", "analysis_type"]
        }
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        """Process the analysis request."""
        inputs = message.inputs or {}
        data = inputs.get('data', '')
        analysis_type = inputs.get('analysis_type', 'summary')
        threshold = inputs.get('confidence_threshold', 0.8)
        
        # Simulate analysis
        result = {
            'analysis_type': analysis_type,
            'data_length': len(data),
            'confidence': threshold,
            'result': f"Analyzed '{data[:50]}...' with {analysis_type} analysis"
        }
        
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=result,
            metadata={'analysis_type': analysis_type}
        )


class WriterAgent(Agent):
    """Agent with simple tool schema for writing."""
    
    def _get_agent_input_schema(self) -> dict[str, any]:
        """Custom schema for writing tasks."""
        return {
            "type": "object", 
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to write about"
                },
                "style": {
                    "type": "string",
                    "enum": ["formal", "casual", "technical"],
                    "description": "Writing style"
                },
                "length": {
                    "type": "integer",
                    "description": "Target length in words"
                }
            },
            "required": ["topic"]
        }
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        """Process the writing request."""
        inputs = message.inputs or {}
        topic = inputs.get('topic', 'General topic')
        style = inputs.get('style', 'formal')
        length = inputs.get('length', 500)
        
        result = f"A {style} {length}-word piece about {topic}"
        
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=result,
            metadata={'topic': topic, 'style': style}
        )


if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🎯 Custom Tool Schema Demo")
        print("=" * 40)
        
        # Create analyzer agent
        analyzer_config = AgentConfig(
            agent_id='analyzer_agent',
            role='ANALYZER',
            description='Analyzes data with multiple methods',
            parameters={'model': 'gpt-4'},
            tools=[]
        )
        analyzer = AnalyzerAgent(**analyzer_config.model_dump())
        
        # Create writer agent  
        writer_config = AgentConfig(
            agent_id='writer_agent',
            role='WRITER',
            description='Writes content in various styles',
            parameters={'model': 'gpt-4'},
            tools=[]
        )
        writer = WriterAgent(**writer_config.model_dump())
        
        # Show their tool definitions
        print("🔧 Analyzer Tool:")
        analyzer_tool = analyzer.get_autogen_tool_definition()
        print(f"  Name: {analyzer_tool['name']}")
        print(f"  Schema: {analyzer_tool['input_schema']['properties'].keys()}")
        
        print("\n🔧 Writer Tool:")
        writer_tool = writer.get_autogen_tool_definition()
        print(f"  Name: {writer_tool['name']}")
        print(f"  Schema: {writer_tool['input_schema']['properties'].keys()}")
        
        # Test the agents
        print("\n🧪 Testing Analyzer:")
        analyzer_input = AgentInput(inputs={
            'data': 'This is some sample data to analyze',
            'analysis_type': 'sentiment',
            'confidence_threshold': 0.9
        })
        analyzer_result = await analyzer._process(message=analyzer_input)
        print(f"  Result: {analyzer_result.outputs}")
        
        print("\n🧪 Testing Writer:")
        writer_input = AgentInput(inputs={
            'topic': 'Artificial Intelligence',
            'style': 'technical',
            'length': 1000
        })
        writer_result = await writer._process(message=writer_input)
        print(f"  Result: {writer_result.outputs}")
    
    asyncio.run(demo())