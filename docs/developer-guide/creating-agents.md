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

### Agent with Configuration

```python
from pydantic import BaseModel, Field
from typing import Optional

class ContentAnalyzerConfig(BaseModel):
    """Configuration for content analysis agent."""
    model: str = Field(default="gemini-pro", description="LLM model to use")
    temperature: float = Field(default=0.7, description="Generation temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    analysis_type: str = Field(default="sentiment", description="Type of analysis")

class ContentAnalyzer(Agent):
    def __init__(self, analysis_type: str = "sentiment", **kwargs):
        super().__init__(**kwargs)
        self.analysis_type = analysis_type
        self.llm_client = self._create_llm_client()
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        prompt = self._build_prompt(record.content)
        response = await self.llm_client.generate(prompt)
        
        return AgentOutput(
            agent_name=self.name,
            content=response,
            metadata={
                "analysis_type": self.analysis_type,
                "model": self.config.get("model", "unknown")
            }
        )
    
    def _build_prompt(self, content: str) -> str:
        if self.analysis_type == "sentiment":
            return f"Analyze the sentiment of this text: {content}"
        elif self.analysis_type == "bias":
            return f"Identify potential bias in this text: {content}"
        else:
            return f"Analyze this text: {content}"
```

## Agent Types

### 1. LLM Agent

For text processing using language models:

```python
from buttermilk.agents.llm import LLMAgent

class CustomLLMAgent(LLMAgent):
    def __init__(self, system_prompt: str = None, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt or "You are a helpful assistant."
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        # Use the base LLM functionality
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": record.content}
        ]
        
        response = await self.llm_client.generate(messages)
        
        return AgentOutput(
            agent_name=self.name,
            content=response,
            metadata={
                "model": self.model,
                "system_prompt": self.system_prompt
            }
        )
```

### 2. RAG Agent

For retrieval-augmented generation:

```python
from buttermilk.agents.rag import RAGAgent

class PolicyRAGAgent(RAGAgent):
    def __init__(self, knowledge_base: str = None, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base or "default"
        self.retriever = self._create_retriever()
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        # Retrieve relevant context
        context = await self.retriever.retrieve(record.content)
        
        # Build prompt with context
        prompt = self._build_rag_prompt(record.content, context)
        
        # Generate response
        response = await self.llm_client.generate(prompt)
        
        return AgentOutput(
            agent_name=self.name,
            content=response,
            metadata={
                "knowledge_base": self.knowledge_base,
                "retrieved_docs": len(context),
                "context_tokens": sum(len(doc.content) for doc in context)
            }
        )
    
    def _build_rag_prompt(self, query: str, context: list) -> str:
        context_text = "\n".join([doc.content for doc in context])
        return f"""
        Context: {context_text}
        
        Query: {query}
        
        Answer the query based on the provided context.
        """
```

### 3. Multi-Modal Agent

For processing multiple types of content:

```python
from buttermilk.agents.multimodal import MultiModalAgent

class ImageTextAgent(MultiModalAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision_client = self._create_vision_client()
        self.text_client = self._create_text_client()
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        results = {}
        
        # Process image content
        if record.image_url:
            image_analysis = await self.vision_client.analyze(record.image_url)
            results["image_analysis"] = image_analysis
        
        # Process text content
        if record.content:
            text_analysis = await self.text_client.analyze(record.content)
            results["text_analysis"] = text_analysis
        
        # Combine results
        combined_analysis = self._combine_analyses(results)
        
        return AgentOutput(
            agent_name=self.name,
            content=combined_analysis,
            metadata={
                "modalities": list(results.keys()),
                "image_processed": bool(record.image_url),
                "text_processed": bool(record.content)
            }
        )
```

### 4. Tool-Enabled Agent

For agents that can use external tools:

```python
from buttermilk._core.mcp_decorators import tool
from buttermilk._core.tool_definition import AgentToolDefinition

class ToolEnabledAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calculator = Calculator()
        self.web_search = WebSearch()
    
    @tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "expression": {"type": "string", "description": "Mathematical expression"}
        }
    )
    async def calculate(self, expression: str) -> dict:
        """Calculate a mathematical expression."""
        try:
            result = self.calculator.evaluate(expression)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}
    
    @tool(
        name="search_web",
        description="Search the web for information",
        parameters={
            "query": {"type": "string", "description": "Search query"}
        }
    )
    async def search_web(self, query: str) -> dict:
        """Search the web for information."""
        results = await self.web_search.search(query)
        return {
            "query": query,
            "results": results[:5],  # Top 5 results
            "total_found": len(results)
        }
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        # This agent can use tools during processing
        content = record.content
        
        # Example: Use tools if needed
        if "calculate" in content.lower():
            # Extract expression and calculate
            expr = self._extract_expression(content)
            calc_result = await self.calculate(expr)
            response = f"Calculation result: {calc_result}"
        elif "search" in content.lower():
            # Extract query and search
            query = self._extract_query(content)
            search_result = await self.search_web(query)
            response = f"Search results: {search_result}"
        else:
            response = f"Processed: {content}"
        
        return AgentOutput(
            agent_name=self.name,
            content=response,
            metadata={"tools_used": ["calculate", "search_web"]}
        )
```

## Configuration

### Agent Configuration Files

```yaml
# conf/agents/my_agent.yaml
name: "content_analyzer"
type: "ContentAnalyzer"
role: "ANALYZER"
description: "Analyzes content for sentiment and bias"

parameters:
  model: "gemini-pro"
  temperature: 0.7
  max_tokens: 1000
  analysis_type: "sentiment"

tools:
  - name: "sentiment_analysis"
    enabled: true
  - name: "bias_detection"
    enabled: true

metadata:
  version: "1.0.0"
  author: "Your Name"
  created_date: "2024-01-15"
```

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

## Advanced Patterns

### Stateful Agent

```python
class StatefulAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation_history = []
        self.user_preferences = {}
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        # Update conversation history
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "content": record.content,
            "record_id": record.id
        })
        
        # Use history in processing
        context = self._build_context_from_history()
        response = await self._generate_response(record.content, context)
        
        return AgentOutput(
            agent_name=self.name,
            content=response,
            metadata={
                "conversation_length": len(self.conversation_history),
                "context_used": bool(context)
            }
        )
    
    def _build_context_from_history(self) -> str:
        if len(self.conversation_history) <= 1:
            return ""
        
        recent_history = self.conversation_history[-5:]  # Last 5 interactions
        return "\n".join([item["content"] for item in recent_history])
```

### Batch Processing Agent

```python
class BatchProcessingAgent(Agent):
    def __init__(self, batch_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.current_batch = []
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        # Add to current batch
        self.current_batch.append(record)
        
        # Process batch when full
        if len(self.current_batch) >= self.batch_size:
            batch_results = await self._process_batch(self.current_batch)
            self.current_batch = []
            
            return AgentOutput(
                agent_name=self.name,
                content=batch_results,
                metadata={
                    "batch_size": self.batch_size,
                    "records_processed": len(batch_results)
                }
            )
        
        # Return placeholder for incomplete batch
        return AgentOutput(
            agent_name=self.name,
            content="Batch incomplete",
            metadata={"batch_progress": len(self.current_batch)}
        )
    
    async def _process_batch(self, records: list[Record]) -> list[dict]:
        # Process all records in batch
        tasks = [self._process_single_record(record) for record in records]
        return await asyncio.gather(*tasks)
```

### Agent with External APIs

```python
import aiohttp
from typing import Optional

class ExternalAPIAgent(Agent):
    def __init__(self, api_key: str, base_url: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _setup(self):
        """Setup external resources."""
        self.session = aiohttp.ClientSession()
    
    async def _cleanup(self):
        """Cleanup external resources."""
        if self.session:
            await self.session.close()
    
    async def _process(self, *, record: Record, **kwargs) -> AgentOutput:
        # Call external API
        response = await self._call_api(record.content)
        
        return AgentOutput(
            agent_name=self.name,
            content=response,
            metadata={
                "api_endpoint": f"{self.base_url}/analyze",
                "api_response_time": response.get("processing_time", 0)
            }
        )
    
    async def _call_api(self, content: str) -> dict:
        """Call external API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"text": content}
        
        async with self.session.post(
            f"{self.base_url}/analyze",
            json=payload,
            headers=headers
        ) as response:
            return await response.json()
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

**Stateless Processing:**
- Agents should be stateless by default
- State should be explicit and managed carefully

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