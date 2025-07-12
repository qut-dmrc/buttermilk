# Your First Flow

This tutorial will guide you through creating your first custom Buttermilk flow from scratch. By the end, you'll understand how flows work and how to build your own.

## Prerequisites

- Buttermilk is [installed](installation.md) and configured
- You've completed the [Quick Start Guide](quickstart.md)
- Basic understanding of YAML configuration files

## What We'll Build

We'll create a simple sentiment analysis flow that:
1. Takes text input
2. Analyzes sentiment using an LLM
3. Returns structured results

## Step 1: Understanding Flow Structure

Every Buttermilk flow has three main components:

```yaml
# Basic flow structure
flow:
  name: my_flow
  description: "What this flow does"
  
agents:
  - name: analyzer
    type: LLMAgent
    # agent configuration
    
data:
  source: inline
  # data configuration
```

## Step 2: Create Your Flow Configuration

Create a new file `conf/flows/sentiment_tutorial.yaml`:

```yaml
# conf/flows/sentiment_tutorial.yaml
defaults:
  - /agents/llm: gemini_simple
  - _self_

flow:
  name: sentiment_tutorial
  description: "Tutorial flow for sentiment analysis"
  version: "1.0.0"
  
agents:
  - name: sentiment_analyzer
    type: LLMAgent
    model: gemini-pro
    system_prompt: |
      You are a sentiment analysis expert. Analyze the given text and return:
      1. Overall sentiment (positive, negative, neutral)
      2. Confidence score (0-1)
      3. Key phrases that influenced your decision
      
      Respond in JSON format:
      {
        "sentiment": "positive|negative|neutral",
        "confidence": 0.85,
        "key_phrases": ["phrase1", "phrase2"],
        "explanation": "Brief explanation of reasoning"
      }
    
data:
  source: inline
  records:
    - id: "sample_1"
      text: "I love this new feature! It's incredibly useful."
    - id: "sample_2"
      text: "This is terrible. I hate how complicated it is."
    - id: "sample_3"
      text: "The weather is okay today, nothing special."

orchestrator:
  type: SequenceOrchestrator
  sequence:
    - sentiment_analyzer
```

## Step 3: Test Your Flow

### Console Mode Test

```bash
# Test the flow
uv run python -m buttermilk.runner.cli run=console flow=sentiment_tutorial

# Test with custom text
uv run python -m buttermilk.runner.cli run=console flow=sentiment_tutorial +prompt="I'm really excited about this new project!"
```

### API Mode Test

```bash
# Start API server
uv run python -m buttermilk.runner.cli run=api flow=sentiment_tutorial

# Test with curl
curl -X POST http://localhost:8000/flow/sentiment_tutorial \
  -H "Content-Type: application/json" \
  -d '{
    "flow": "sentiment_tutorial",
    "prompt": "This tutorial is really helpful!",
    "text": "I learned so much about Buttermilk flows."
  }'
```

## Step 4: Understanding What Happened

When you run the flow, Buttermilk:

1. **Loads Configuration**: Reads your YAML file and resolves defaults
2. **Creates Agents**: Instantiates the LLM agent with your prompt
3. **Processes Data**: Feeds each record through the agent
4. **Returns Results**: Structured output with sentiment analysis

## Step 5: Customize Your Flow

### Add Multiple Agents

```yaml
# Extended version with multiple agents
agents:
  - name: preprocessor
    type: LLMAgent
    model: gemini-pro
    system_prompt: "Clean and normalize the input text. Remove unnecessary whitespace and fix obvious typos."
    
  - name: sentiment_analyzer
    type: LLMAgent
    model: gemini-pro
    system_prompt: |
      Analyze sentiment of the preprocessed text...
      
  - name: confidence_checker
    type: LLMAgent
    model: gemini-pro
    system_prompt: "Review the sentiment analysis and provide a confidence assessment."

orchestrator:
  type: SequenceOrchestrator
  sequence:
    - preprocessor
    - sentiment_analyzer
    - confidence_checker
```

### Add Data Sources

```yaml
# Use different data sources
data:
  source: csv
  path: "/path/to/your/data.csv"
  text_column: "content"
  id_column: "id"
  
# Or use Google Sheets
data:
  source: gsheet
  spreadsheet_id: "your-sheet-id"
  range: "Sheet1!A:B"
```

### Add Storage Configuration

```yaml
# Store results in BigQuery
storage:
  type: bigquery
  project: "your-project"
  dataset: "sentiment_analysis"
  table: "results"
  
# Or store locally
storage:
  type: local
  path: "/tmp/sentiment_results"
  format: "json"
```

## Step 6: Advanced Features

### Error Handling

```yaml
agents:
  - name: sentiment_analyzer
    type: LLMAgent
    model: gemini-pro
    retry_count: 3
    timeout: 30
    fallback_model: gemini-flash
    system_prompt: "Your prompt here..."
```

### Conditional Logic

```yaml
orchestrator:
  type: LLMHostOrchestrator
  model: gemini-pro
  system_prompt: |
    You coordinate sentiment analysis. Based on the input:
    - For short texts (< 100 chars): use quick_analyzer
    - For long texts: use detailed_analyzer
    - For unclear inputs: use both and compare
    
  available_agents:
    - quick_analyzer
    - detailed_analyzer
```

### Human-in-the-Loop

```yaml
agents:
  - name: human_reviewer
    type: HumanAgent
    prompt: "Please review this sentiment analysis. Is it accurate?"
    
orchestrator:
  type: SequenceOrchestrator
  sequence:
    - sentiment_analyzer
    - human_reviewer  # Pauses for human input
```

## Step 7: Production Considerations

### Environment Configuration

Create `conf/local.yaml` for local development:

```yaml
# conf/local.yaml
defaults:
  - _self_

llms:
  gemini:
    project: "your-dev-project"
    
logging:
  level: DEBUG
  
storage:
  type: local
  path: "/tmp/dev-results"
```

### Batch Processing

```bash
# Process large datasets
uv run python -m buttermilk.runner.cli run=batch flow=sentiment_tutorial \
  data.source=csv \
  data.path=/path/to/large_dataset.csv \
  run.human_in_loop=false
```

## Step 8: Testing Your Flow

Create a test file `tests/test_sentiment_tutorial.py`:

```python
import pytest
from buttermilk.runner.cli import main

def test_sentiment_tutorial():
    """Test the sentiment tutorial flow."""
    # Test configuration loading
    config = main.load_config("sentiment_tutorial")
    assert config.flow.name == "sentiment_tutorial"
    
    # Test flow execution
    results = main.run_flow(config, test_mode=True)
    assert len(results) > 0
    assert "sentiment" in results[0]
```

## Next Steps

Congratulations! You've created your first custom Buttermilk flow. Here's what to explore next:

### Learn More
- [Configuration Guide](../user-guide/configuration.md) - Deep dive into Hydra configuration
- [Creating Agents](../developer-guide/creating-agents.md) - Build custom agent types
- [API Reference](../user-guide/api-reference.md) - Use your flows via HTTP API

### Advanced Topics
- [Architecture](../developer-guide/architecture.md) - Understand the system design
- [Testing](../developer-guide/testing.md) - Test your flows thoroughly
- [Troubleshooting](../reference/troubleshooting.md) - Debug common issues

### Community
- Share your flow with the community
- Contribute to the [main repository](https://github.com/qut-dmrc/buttermilk)
- Join discussions about HASS computational methods

## Common Patterns

### Text Analysis Flow
```yaml
# Good for: Content analysis, classification, extraction
agents:
  - name: classifier
  - name: extractor
  - name: summarizer
```

### Multi-Modal Flow
```yaml
# Good for: Image + text analysis, video processing
agents:
  - name: image_analyzer
  - name: text_analyzer
  - name: fusion_agent
```

### Validation Flow
```yaml
# Good for: Quality checking, fact verification
agents:
  - name: primary_analyzer
  - name: validator
  - name: human_reviewer
```

## Tips for Success

1. **Start Simple**: Begin with single-agent flows
2. **Test Early**: Use console mode for quick testing
3. **Iterate**: Refine prompts based on results
4. **Document**: Add clear descriptions to your flows
5. **Version**: Track changes to your configurations
6. **Monitor**: Watch for errors and performance issues

Remember: The best flow is one that solves your specific research problem effectively and reliably!