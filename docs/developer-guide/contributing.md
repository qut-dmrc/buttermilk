# Contributing Guide

Welcome to the Buttermilk project! This guide provides essential information for contributing to our HASS-researcher-friendly AI framework.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Common Patterns](#common-patterns)
- [Getting Help](#getting-help)

## Getting Started

### Prerequisites
- Python 3.10+ (3.12 recommended)
- Git
- Google Cloud SDK (for authentication)
- uv package manager

### Initial Setup
1. Fork the repository
2. Clone your fork locally
3. Install dependencies: `uv install`
4. Set up authentication: `gcloud auth login`
5. Run tests: `uv run pytest`

### Required Reading
Before making any changes, please read:
- [Architecture Documentation](architecture.md)
- [Core Concepts](../reference/concepts.md)
- [Configuration Guide](../user-guide/configuration.md)

## Development Process

For detailed development methodology and best practices, see the `CLAUDE.md` file in the repository root, which contains essential guidance for systematic development.

## Code Standards

### Architecture Patterns

**Use Existing Base Classes:**
- Prefer `AgentTrace` and `AgentOutputs` over custom objects
- Store data in Buttermilk base classes for consistency
- Avoid creating separate objects for specific UI needs

**Modular Design:**
- Create new Agent/Orchestrator subclasses over modifying core components
- Use composition over inheritance where possible
- Keep interfaces stable

### Configuration Management

**Hydra-First Approach:**
- Use OmegaConf objects exclusively
- No manual dictionary configuration
- Prefer YAML configuration files

**Example:**
```python
from omegaconf import OmegaConf

# Good
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    bm = BM(cfg.bm)
    
# Bad
def main():
    config = {"model": "gpt-4"}  # Manual dictionary
```

### Async Operations

**Embrace Async/Await:**
- Use async/await for I/O-bound tasks
- Make LLM calls asynchronous
- Ensure responsive concurrent operations

**Example:**
```python
# Good
async def analyze_content(self, content: str) -> str:
    response = await self.llm_client.generate(content)
    return response

# Bad
def analyze_content(self, content: str) -> str:
    response = self.llm_client.generate(content)  # Blocking call
    return response
```

### Data Handling

**Use Pydantic for Validation:**
- Define data models with Pydantic v2
- Use validators for complex validation logic
- Avoid validation in main methods

**Example:**
```python
from pydantic import BaseModel, validator

class AgentConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Temperature must be between 0 and 1')
        return v
```

### Error Handling

**Fail Fast Philosophy:**
- Use validators on inputs
- Let errors percolate up
- Don't create defensive code everywhere

**Example:**
```python
# Good
async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
    # Validation happens at model level
    return await self.agent.process(message)

# Bad
async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
    if not message:
        return None
    if not hasattr(message, 'content'):
        return None
    # Lots of defensive checks...
```

## Testing Guidelines

### Test-Driven Development

**Write Tests First:**
1. Write a failing test that demonstrates the problem
2. Implement the minimal code to make it pass
3. Refactor while keeping tests green

**Test Structure:**
```python
def test_agent_processes_content():
    # Arrange
    agent = MyAgent(config)
    test_content = "Test content"
    
    # Act
    result = agent.process(test_content)
    
    # Assert
    assert result.status == "completed"
    assert "analysis" in result.content
```

### Test Categories

**Unit Tests:**
- Test individual functions/methods
- Mock external dependencies
- Fast execution (< 1 second each)

**Integration Tests:**
- Test component interactions
- Use real dependencies where possible
- Test end-to-end workflows

**Example:**
```python
# Unit test
def test_prompt_template_formatting():
    template = PromptTemplate("Analyze: {text}")
    result = template.format(text="sample")
    assert result == "Analyze: sample"

# Integration test
def test_full_flow_execution():
    flow = Flow.from_config("test_flow")
    result = flow.run({"text": "test content"})
    assert result.success
```

### Authentication in Tests

**GCP Authentication Required:**
- Tests need GCP authentication for LLM access
- Use service account or user credentials
- Tests will fail without proper authentication

**Setup:**
```bash
# Authenticate before running tests
gcloud auth login
gcloud auth application-default login
uv run pytest
```

## Documentation Standards

### Code Documentation

**Docstrings (Google Style):**
```python
def process_content(self, content: str, model: str = "gpt-4") -> ProcessingResult:
    """Process content using the specified model.
    
    Args:
        content: The text content to process
        model: The LLM model to use for processing
        
    Returns:
        ProcessingResult containing the analysis and metadata
        
    Raises:
        ValidationError: If content is empty or invalid
        ModelError: If the specified model is not available
    """
```

**Class Documentation:**
```python
class ContentAnalyzer:
    """Analyzes content for sentiment, bias, and other metrics.
    
    This class provides methods for analyzing text content using various
    LLM models. It handles prompt templating, result parsing, and error
    handling.
    
    Attributes:
        config: Configuration object containing model settings
        llm_client: Client for interacting with LLM APIs
        
    Example:
        >>> analyzer = ContentAnalyzer(config)
        >>> result = analyzer.analyze("Sample text")
        >>> print(result.sentiment)
        "positive"
    """
```

### Markdown Documentation

**Clear Structure:**
- Use headers to organize content
- Include code examples
- Provide links to related documentation

**Update When Changing:**
- Keep docs in sync with code changes
- Update examples when APIs change
- Add new features to relevant guides

## Common Patterns

### Agent Development

**Base Agent Structure:**
```python
class MyAgent(Agent):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.llm = self._create_llm_client()
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        # Process the message
        result = await self.llm.generate(message.content)
        return AgentOutput(
            content=result,
            metadata={"tokens": len(result)}
        )
```

### Flow Configuration

**Hierarchical Configuration:**
```yaml
# conf/flows/my_flow.yaml
defaults:
  - _self_
  - /agents/llm: gemini_simple
  - /data: local_files

flow:
  name: my_flow
  description: "Custom analysis flow"
  
agents:
  - name: analyzer
    type: LLMAgent
    prompt: "Analyze this content"
```

### Error Handling

**Structured Error Responses:**
```python
class ProcessingError(Exception):
    """Base exception for processing errors."""
    
    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
```

## Git and GitHub Workflow

### Commit Standards

**Commit Message Format:**
```
type: short description

Longer explanation if needed. Explain WHY the change was made,
not just what was changed.

Fixes #123
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### GitHub Issues

**Before Committing:**
1. Check for related issues: `gh issue list --search "keywords"`
2. Create issue if none exists: `gh issue create`
3. Make your changes
4. Update the issue with progress

**After Committing:**
```bash
# Update issue with commit
gh issue comment <issue-number> --body "Fixed in commit abc123"

# Close issue if complete
gh issue close <issue-number>
```

### Pull Request Process

1. **Create Feature Branch:** `git checkout -b feature/my-feature`
2. **Make Changes:** Follow the development process
3. **Update Documentation:** Keep docs in sync
4. **Run Tests:** Ensure all tests pass
5. **Create PR:** Use the provided template
6. **Address Review:** Respond to feedback promptly

## Development Anti-Patterns to Avoid

### Red Flags

Stop immediately if you find yourself:
- Proposing config changes without understanding data flow
- Making multiple small fixes instead of addressing root cause
- Suggesting "try this" without a systematic plan
- Modifying code without tests that demonstrate the problem
- Changing Pydantic validation settings to suppress errors

### Never Do These

**Superficial Fixes:**
- Changing `extra="forbid"` to `extra="allow"` in Pydantic models
- Adding try/except blocks to suppress validation errors
- Using `# type: ignore` without understanding why
- Making type annotations more permissive to bypass checking

**When You Encounter Validation Errors:**
1. Trace the data flow (YAML → Hydra → Pydantic)
2. Identify the proper conversion point
3. Fix at the right level
4. Test the fix thoroughly

## Development Environment

### Running Commands

**Always Use uv:**
```bash
# Run Python
uv run python script.py

# Run tests
uv run pytest

# Run CLI
uv run python -m buttermilk.runner.cli
```

### Key Components

**BM Class:**
- Critical component for all buttermilk projects
- Handles logging and authentication
- Provides MLOps best practices
- Access with `get_bm()`, initialize with `set_bm()`

**Configuration Inspection:**
```bash
# View compiled configuration
uv run python -m buttermilk.runner.cli -c job

# Check specific flow
uv run python -m buttermilk.runner.cli flow=my_flow -c job
```

### Memory and Performance

**Best Practices:**
- Use generators for large datasets
- Implement proper cleanup in agents
- Monitor memory usage in batch processing
- Use async operations for I/O

## Getting Help

### Resources

**Documentation:**
- [Architecture Guide](architecture.md)
- [Testing Guide](testing.md)
- [API Reference](../user-guide/api-reference.md)
- [Configuration Guide](../user-guide/configuration.md)

**Community:**
- GitHub Discussions for questions
- GitHub Issues for bugs/features
- Email: [nic@suzor.com](mailto:nic@suzor.com)

### Common Issues

**Authentication Problems:**
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

**Configuration Issues:**
```bash
# Check configuration
uv run python -m buttermilk.runner.cli --cfg job

# List available options
uv run python -m buttermilk.runner.cli --info searchpath
```

**Test Failures:**
```bash
# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_specific.py::test_function
```

## Quality Checklist

Before submitting a PR, ensure:

- [ ] Analysis phase completed
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] GitHub issue updated
- [ ] No regressions introduced
- [ ] Edge cases handled

## Contributing Examples

### Adding a New Agent

1. **Analysis:** Understand the agent's purpose and interface
2. **Planning:** Create GitHub issue with implementation plan
3. **Testing:** Write tests for the agent behavior
4. **Implementation:** Create the agent class
5. **Documentation:** Add docstrings and usage examples
6. **Integration:** Add configuration files and examples

### Fixing a Bug

1. **Analysis:** Reproduce the bug and identify root cause
2. **Planning:** Create focused fix plan
3. **Testing:** Write test that fails with current code
4. **Implementation:** Fix the bug
5. **Validation:** Ensure test passes and no regressions
6. **Documentation:** Update relevant docs if needed

## Conclusion

Contributing to Buttermilk requires discipline and systematic thinking, but this approach ensures high-quality, maintainable code that serves our HASS research community effectively. When in doubt, prioritize understanding over speed, and don't hesitate to ask for help.

Remember: It's better to say "Let me analyze this systematically" than to guess. Unknown unknowns are the most dangerous - surface them early through systematic analysis.

Thank you for contributing to Buttermilk!