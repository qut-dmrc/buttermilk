# Tool Descriptions Guide

## Overview

Good tool descriptions help LLMs choose the right agent for each task. The system now automatically enhances descriptions to explain **when** to use tools, not just **what** they do.

## How Tool Descriptions Work

### 🔄 Description Flow
```
Agent Config → Enhanced Description → Tool Definition → LLM Host → LLM Selection
```

### 🎯 Three Ways to Set Descriptions

#### 1. **Custom Override Method** (Most Control)
```python
class DataAnalyzer(Agent):
    def _get_tool_description(self) -> str:
        return (
            "Use this tool when you need to analyze, process, or extract insights from data. "
            "Ideal for: statistical analysis, data cleaning, pattern detection, trend identification, "
            "data visualization recommendations, and quantitative research tasks. "
            "Input should include the data to analyze and the type of analysis required."
        )
```

#### 2. **Parameter-Based** (Configuration)
```python
config = AgentConfig(
    role='WRITER',
    description='Creates written content',
    parameters={
        'model': 'gpt-4',
        'tool_description': (
            "Use this tool when you need to create written content such as articles, "
            "blog posts, documentation, marketing copy, or creative writing. "
            "Specify the topic, target audience, tone, and desired length."
        )
    }
)
```

#### 3. **Auto-Enhanced** (From agent description)
```python
config = AgentConfig(
    role='LEGAL',
    description='Review contracts, assess legal risks, and provide compliance guidance',
    # Becomes: "Use this tool when you need to: review contracts, assess legal risks, 
    # and provide compliance guidance. Calls the LEGAL agent to handle legal-specific tasks."
)
```

## Best Practices

### ✅ Good Tool Descriptions
- **Explain WHEN to use the tool**
- **List specific use cases**
- **Mention input requirements**
- **Use action-oriented language**

#### Examples:
```
✅ "Use this tool when you need to analyze data for patterns, trends, or statistical insights. 
    Ideal for: data cleaning, exploratory analysis, visualization planning, and research tasks."

✅ "Use this tool to search for and retrieve information from knowledge bases, documents, 
    or databases. Best for: fact-checking, research queries, and finding specific information."

✅ "Use this tool when you need to generate, review, or modify code in any programming language. 
    Suitable for: debugging, code review, refactoring, and implementation tasks."
```

### ❌ Poor Tool Descriptions
```
❌ "Analyzes data"  (Too vague)
❌ "Helper agent"   (No context)
❌ "Does stuff"     (Meaningless)
```

## Configuration Examples

### YAML Configuration
```yaml
# agents/analyzer.yaml
agent_id: data_analyzer
role: ANALYZER
description: Performs statistical analysis, data cleaning, and pattern detection
parameters:
  model: gpt-4
  tool_description: >
    Use this tool when you need to analyze data for insights, patterns, or statistical information.
    Ideal for: data exploration, statistical analysis, trend identification, data quality assessment,
    and preparing data for visualization. Provide the data and specify the type of analysis needed.
```

### Python Configuration
```python
analyzer_config = AgentConfig(
    agent_id='data_analyzer',
    role='ANALYZER',
    description='Performs statistical analysis, data cleaning, and pattern detection',
    parameters={
        'model': 'gpt-4',
        'tool_description': (
            "Use this tool when you need to analyze data for insights, patterns, or statistical information. "
            "Ideal for: data exploration, statistical analysis, trend identification, data quality assessment, "
            "and preparing data for visualization. Provide the data and specify the type of analysis needed."
        )
    }
)
```

## Testing Tool Descriptions

### Quick Test
```python
from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig

config = AgentConfig(
    role='TESTER',
    description='Your agent description here',
    parameters={'tool_description': 'Your custom tool description'},
    tools=[]
)

class TestAgent(Agent):
    async def _process(self, *, message, **kwargs):
        return {'result': 'test'}

agent = TestAgent(**config.model_dump())
tool_def = agent.get_autogen_tool_definition()
print(f"Tool description: {tool_def['description']}")
```

### In Host Environment
The enhanced descriptions automatically flow to the LLM host:
```python
# Host automatically gets enhanced descriptions
host = StructuredLLMHostAgent(...)
# Agent announces with enhanced tool definition
agent_announcement = agent.create_announcement('initial', 'joining')
# Host builds tools with rich descriptions
await host.update_agent_registry(agent_announcement)
# LLM receives detailed tool descriptions for better selection
```

## Advanced Customization

### Role-Specific Descriptions
```python
class SpecializedAgent(Agent):
    def _get_tool_description(self) -> str:
        role_descriptions = {
            'RESEARCHER': "Use for finding, verifying, and synthesizing information from multiple sources...",
            'ANALYZER': "Use for statistical analysis, data processing, and extracting insights...",
            'WRITER': "Use for creating, editing, and formatting written content...",
            'CODER': "Use for writing, reviewing, debugging, and optimizing code..."
        }
        
        base_desc = role_descriptions.get(self.role, f"Use for {self.role.lower()}-related tasks")
        
        # Add context-specific guidance
        if self.parameters.get('domain'):
            domain = self.parameters['domain']
            base_desc += f" Specialized in {domain} domain."
            
        return base_desc
```

### Context-Aware Descriptions
```python
def _get_tool_description(self) -> str:
    # Base description
    desc = "Use this tool for data analysis tasks including..."
    
    # Add capability info
    if self.parameters.get('supports_visualization'):
        desc += " Supports data visualization recommendations."
    
    if self.parameters.get('real_time_data'):
        desc += " Can process real-time data streams."
        
    return desc
```

## Impact on LLM Performance

### Before Enhancement
```
Tool: call_analyzer
Description: "Analyzes data"
→ LLM often confused about when to use
```

### After Enhancement  
```
Tool: call_analyzer  
Description: "Use this tool when you need to analyze data for patterns, trends, or statistical insights..."
→ LLM makes better tool selection decisions
```

## Summary

✅ **Enhanced descriptions are automatic** - all agents get them  
✅ **Three customization levels** - override method, parameters, or auto-enhanced  
✅ **Better LLM tool selection** - clear guidance on when to use each tool  
✅ **Flows through announcement system** - automatically reaches host agents  
✅ **Backward compatible** - existing agents work with improvements  

The system now provides rich, contextual tool descriptions that guide LLMs to make better tool selection decisions! 🎯