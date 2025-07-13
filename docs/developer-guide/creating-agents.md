# Creating Agents

This guide covers how to create custom agents in Buttermilk, from basic agents to complex multi-modal processors.

## Table of Contents
- [Agent Overview](#agent-overview)
- [Basic Agent Structure](#basic-agent-structure)
- [Agent Types](#agent-types)
- [Configuration](#configuration)
- [Tool Integration](#tool-integration)
- [Advanced Patterns](#advanced-patterns)
- [Testing Agents](#testing-agents)
- [Best Practices](#best-practices)

## Agent Overview

Agents are the core processing units in Buttermilk. They:
- Process individual records
- Maintain state and context
- Generate structured outputs
- Support tool calling and external integrations
- Provide rich metadata and tracing

### Agent Lifecycle

1. **Initialization**: Agent is created with configuration
2. **Setup**: Agent prepares resources and connections
3. **Processing**: Agent processes records one by one
4. **Cleanup**: Agent releases resources
5. **Reset**: Agent can be reset for reuse

## Basic Agent Structure

### Minimal Agent

```python
from buttermilk._core.agent import Agent
from buttermilk._core.types import Record, AgentOutput
from omegaconf import DictConfig

class SimpleAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize agent-specific properties
        self.processor = self._create_processor()
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        """Process a single record.
        
        Args:
            record: The record to process
            **kwargs: Additional context
            
        Returns:
            AgentOutput with processing results
        """
        # Process the record
        result = await self.processor.process(record.content)
        
        # Return structured output
        return AgentOutput(
            agent_name=self.name,
            content=result,
            metadata={
                "processing_time": 1.23,
                "tokens_used": 150
            }
        )
    
    def _create_processor(self):
        """Create the processor for this agent."""
        # Implementation depends on agent type
        pass
```

## Real Example Agents

Instead of dummy examples, refer to the working example agents and flows in the repository:

**Example Flows:**
- `conf/flows/trans.yaml` - Journalism quality assessment for trans issues reporting
- `conf/flows/osb.yaml` - Interactive group chat for querying OSB vector store  
- `conf/flows/tox.yaml` - Toxicity criteria application
- `conf/flows/zot.yaml` - Zotero RAG for academic citations

**Example Agents:**
- `conf/agents/judge.yaml` - Expert content assessment agent
- `conf/agents/synth.yaml` - Response synthesis agent
- `conf/agents/differences.yaml` - Analysis comparison agent
- `conf/agents/spy.yaml` - Process monitoring agent
- `conf/agents/rag.yaml` - Retrieval-augmented generation agent

**Host Agents:**
- `conf/agents/host/` - Various orchestration agents including sequence_host and llm_host

These configurations demonstrate real, working patterns that have been tested and validated in production. Study these examples to understand:
- How models are configured at the flow level
- Real agent parameter patterns
- Working tool integrations
- Actual input/output specifications

All agent configurations use Hydra's composition system and follow the established patterns in the codebase.

## Configuration

### Agent Configuration Files

Instead of dummy configurations, refer to real working examples:

```yaml
# Real example: conf/agents/judge.yaml
judge:
  role: judge
  description: Expert analysts, particularly suited to assess content with subject matter expertise
  agent_obj: Judge
  name_components: ["⚖️", "role", "model", "criteria", "unique_identifier"]
  num_runs: 1
  parameters:
    template: judge
  variants:
    model: ${llms.judgers}
    criteria: [] 
  inputs:
    records: "FETCH.outputs||*.records[]"
```

```yaml
# Real example: conf/agents/synth.yaml
synthesiser:
  role: SYNTHESISER
  name: "🎨 Synthesiser"
  name_components: ["🎨", "role", "model", "criteria", "unique_identifier"]
  agent_obj: Judge
  description: Team leaders, responsible for synthesizes diverging draft answers
  num_runs: 1
  parameters:
    template: synthesise
    formatting: json_rules
  variants:
    model: ${llms.synthesisers}
    criteria: []
  inputs:
    records: "[FETCH]||*.records[]||*.records[]"
    answers: "[JUDGE][].{agent_id: agent_info.agent_id, agent_name: agent_info.agent_name, result: outputs, answer_id: call_id, error: error }"
```

These examples show how models are configured at the flow level using `${llms.judgers}` and `${llms.synthesisers}` references.

### Using Configuration in Agent

```python
class ConfigurableAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Access configuration parameters
        self.model = self.config.get("model", "gemini-pro")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 1000)
        
        # Initialize based on configuration
        self.llm_client = self._create_llm_client()
    
    def _create_llm_client(self):
        return LLMClient(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
```

## Tool Integration

### Defining Tools with Decorators

```python
from buttermilk._core.mcp_decorators import tool, MCPRoute

class ToolAgent(Agent):
    @tool(
        name="analyze_text",
        description="Analyze text for various attributes",
        parameters={
            "text": {"type": "string", "description": "Text to analyze"},
            "analysis_type": {"type": "string", "enum": ["sentiment", "bias", "toxicity"]}
        }
    )
    async def analyze_text(self, text: str, analysis_type: str = "sentiment") -> dict:
        """Analyze text for specified attributes."""
        # Implementation
        result = await self._perform_analysis(text, analysis_type)
        return {
            "text": text,
            "analysis_type": analysis_type,
            "result": result
        }
    
    @MCPRoute(path="/agent/analyze", method="POST")
    async def mcp_analyze(self, request_data: dict) -> dict:
        """MCP endpoint for text analysis."""
        return await self.analyze_text(
            request_data["text"],
            request_data.get("analysis_type", "sentiment")
        )
```

### Tool Definition Generation

```python
def get_tool_definitions(self) -> list[AgentToolDefinition]:
    """Generate tool definitions for this agent."""
    return [
        AgentToolDefinition(
            name="analyze_text",
            description="Analyze text for various attributes",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                    "analysis_type": {"type": "string", "enum": ["sentiment", "bias"]}
                },
                "required": ["text"]
            },
            mcp_routes=[
                {"path": "/agent/analyze", "method": "POST"}
            ]
        )
    ]
```

## Testing Agents

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from buttermilk._core.types import Record

class TestContentAnalyzer:
    @pytest.fixture
    def agent(self):
        return ContentAnalyzer(
            agent_id="test_agent",
            role="ANALYZER",
            description="Test analyzer",
            analysis_type="sentiment"
        )
    
    @pytest.fixture
    def sample_record(self):
        return Record(
            id="test_1",
            content="This is a test message with positive sentiment."
        )
    
    async def test_agent_processes_record(self, agent, sample_record):
        # Mock LLM client
        with patch.object(agent, 'llm_client') as mock_llm:
            mock_llm.generate.return_value = "positive"
            
            result = await agent._process(record=sample_record)
            
            assert result.agent_name == agent.name
            assert result.content == "positive"
            assert result.metadata["analysis_type"] == "sentiment"
    
    async def test_agent_handles_error(self, agent, sample_record):
        with patch.object(agent, 'llm_client') as mock_llm:
            mock_llm.generate.side_effect = Exception("API Error")
            
            with pytest.raises(Exception, match="API Error"):
                await agent._process(record=sample_record)
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_agent_with_real_llm():
    agent = ContentAnalyzer(
        agent_id="integration_test",
        role="ANALYZER",
        description="Integration test",
        analysis_type="sentiment",
        model="gemini-pro"
    )
    
    record = Record(
        id="test_1",
        content="I love this new feature!"
    )
    
    result = await agent._process(record=record)
    
    assert result.success
    assert result.content
    assert "positive" in result.content.lower()
```

## Best Practices

### 1. Agent Design Principles

**Single Responsibility:**
- Each agent should have one clear purpose
- Avoid creating agents that do too many things

**State Management:**
- Agents in flows may have state and context
- State should be explicit and managed carefully
- Consider whether state is needed for your use case

**Error Handling:**
- Implement proper error handling and recovery
- Provide meaningful error messages

### 2. Configuration Management

**Use Structured Configuration:**
```python
from pydantic import BaseModel

class AgentConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    
    class Config:
        validate_assignment = True
```

**Default Values:**
```python
def __init__(self, model: str = "gemini-pro", **kwargs):
    super().__init__(**kwargs)
    self.model = model
```

### 3. Resource Management

**Async Resource Cleanup:**
```python
async def _cleanup(self):
    """Cleanup resources."""
    if hasattr(self, 'client'):
        await self.client.close()
    if hasattr(self, 'session'):
        await self.session.close()
```

**Connection Pooling:**
```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100)
    )
```

### 4. Monitoring and Observability

**Rich Metadata:**
```python
return AgentOutput(
    agent_name=self.name,
    content=response,
    metadata={
        "processing_time": time.time() - start_time,
        "model": self.model,
        "tokens_used": response.get("usage", {}).get("total_tokens", 0),
        "input_length": len(record.content),
        "output_length": len(response)
    }
)
```

**Performance Tracking:**
```python
import time

async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
    start_time = time.time()
    
    # Processing logic
    result = await self._do_processing(record)
    
    processing_time = time.time() - start_time
    
    return AgentOutput(
        agent_name=self.name,
        content=result,
        metadata={"processing_time": processing_time}
    )
```

### 5. Documentation

**Clear Docstrings:**
```python
class ContentAnalyzer(Agent):
    """Analyzes content for sentiment, bias, and other attributes.
    
    This agent uses large language models to analyze text content
    and provide structured insights about sentiment, potential bias,
    and other relevant attributes.
    
    Attributes:
        analysis_type: Type of analysis to perform
        model: LLM model to use for analysis
        
    Example:
        >>> agent = ContentAnalyzer(analysis_type="sentiment")
        >>> result = await agent.process(record)
        >>> print(result.content)
    """
```

**Configuration Documentation:**
```yaml
# conf/agents/content_analyzer.yaml
# Content Analyzer Agent Configuration
# 
# This agent analyzes text content for various attributes
# including sentiment, bias, and toxicity detection.
#
# Parameters:
#   model: LLM model to use (default: gemini-pro)
#   temperature: Generation temperature (0.0-1.0)
#   analysis_type: Type of analysis (sentiment, bias, toxicity)

name: "content_analyzer"
type: "ContentAnalyzer"
# ... rest of configuration
```

## Conclusion

Creating effective agents in Buttermilk requires understanding the base architecture, following established patterns, and implementing proper error handling and resource management. By following these guidelines, you'll create agents that are:

- Reliable and robust
- Well-integrated with the Buttermilk ecosystem
- Easy to test and maintain
- Scalable and performant

Remember to:
- Start with simple agents and add complexity gradually
- Test thoroughly at both unit and integration levels
- Use proper configuration management
- Implement comprehensive error handling
- Document your agents clearly
- Follow the established patterns and conventions