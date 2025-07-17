# Buttermilk Development Workflow

## 🚨 CRITICAL: Systematic Development Process

The #1 rule: **STOP → ANALYZE → PLAN → TEST → IMPLEMENT → DOCUMENT → COMMIT**

## Development Phases

### Phase 1: STOP - Don't Rush
Before writing ANY code:
- Understand the full problem scope
- Check GitHub issues for related work
- Identify all stakeholders and impacts
- Question assumptions

### Phase 2: ANALYZE - Map the Territory
```bash
# Check existing issues
gh issue list --search "relevant keywords"

# If no issue exists, create one
gh issue create --title "Clear problem description" --body "Detailed analysis"
```

Key Analysis Steps:
1. **Trace Data Flow**: YAML → Hydra → OmegaConf → Pydantic → Agent
2. **Map Dependencies**: What touches what?
3. **Identify Root Cause**: Not symptoms, but actual problems
4. **Document Findings**: Update the GitHub issue

### Phase 3: PLAN - Design Before Code
Create a clear plan with:
- Phases and milestones
- Success criteria
- Test cases
- Rollback strategy

### Phase 4: TEST - Failing Tests First
```python
# Write test that demonstrates the problem
def test_expected_behavior():
    # This should pass when fixed
    agent = MyAgent(config)
    result = agent.process(test_input)
    assert result.status == "expected"
```

### Phase 5: IMPLEMENT - Minimal Changes
- Make the smallest change that fixes root cause
- Don't add "nice to have" features
- Keep existing interfaces stable
- Follow existing patterns

### Phase 6: DOCUMENT - Keep It Current
- Update docstrings
- Add inline comments for complex logic
- Update relevant .md files
- Ensure examples still work

### Phase 7: COMMIT - Track Progress
```bash
# Commit with clear message
git add -A
git commit -m "fix: clear description of what and why

- Detailed explanation
- Reference to issue

Fixes #123"

# Update GitHub issue
gh issue comment 123 --body "Fixed in commit abc123. 
- Changed X to handle Y
- Tests now pass
- No regressions found"
```

## GitHub Workflow

### Before Starting Work
1. **Search Issues**: `gh issue list --search "keywords"`
2. **Read Related**: Check linked issues and PRs
3. **Create/Update Issue**: Document your understanding
4. **Assign Yourself**: `gh issue edit 123 --add-assignee @me`

### During Development
1. **Update Issue Regularly**: Post findings and blockers
2. **Link Commits**: Reference issue in commit messages
3. **Ask Questions**: Comment on issue if stuck
4. **Track Progress**: Use task lists in issue body

### After Completion
1. **Final Update**: Summarize what was done
2. **Close Issue**: `gh issue close 123`
3. **Link PR**: If applicable

## Code Standards

### Python Style
```python
# Good: Clear, typed, documented
async def process_content(
    self, 
    content: str, 
    *, 
    model: str = "gemini-pro"
) -> ProcessingResult:
    """Process content using specified model.
    
    Args:
        content: Text to process
        model: LLM model name
        
    Returns:
        Processing result with metadata
    """
    # Implementation
```

### Configuration Style
```yaml
# Good: Clear structure, documentation
# Agent for analyzing content sentiment
sentiment_analyzer:
  role: ANALYZER
  agent_obj: buttermilk.agents.sentiment.SentimentAgent
  description: "Analyzes text sentiment"
  parameters:
    model: ${llms.general}  # Use interpolation
    threshold: 0.7
```

### Test Style
```python
# Good: Descriptive, isolated, async
@pytest.mark.asyncio
async def test_agent_handles_empty_content():
    """Agent should raise ValueError for empty content."""
    agent = ContentAnalyzer(test_config)
    
    with pytest.raises(ValueError, match="Content cannot be empty"):
        await agent.process("")
```

## Common Patterns

### Creating New Agents
```python
# 1. Inherit from base
class MyAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize resources
        
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        # Core logic
        return AgentOutput(...)
```

### Adding Tools
```python
# Use decorators for tool definition
@tool(name="analyze", description="Analyze text")
async def analyze_text(self, text: str) -> dict:
    # Tool implementation
    return {"result": "analysis"}
```

### Configuration Validation
```python
# Use Pydantic for validation
class MyConfig(BaseModel):
    model: str
    temperature: float = Field(ge=0, le=1)
    
    @validator('model')
    def validate_model(cls, v):
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Unknown model: {v}")
        return v
```

## Anti-Patterns to Avoid

### 🚫 Red Flags - Stop Immediately If You're:
1. Proposing config changes without understanding data flow
2. Making multiple small fixes instead of one root cause fix
3. Suggesting "try this" without a systematic plan
4. Modifying code without tests demonstrating the problem
5. Making "quick fixes" to suppress errors

### 🚫 Never Do These:

#### Superficial Fixes
```python
# NEVER: Change validation to hide errors
class MyModel(BaseModel):
    # BAD: Changed from extra="forbid"
    class Config:
        extra = "allow"  # 🚫 DON'T DO THIS
```

#### Type Suppression
```python
# NEVER: Suppress type errors
result = process_data(data)  # type: ignore  🚫
```

#### Defensive Overload
```python
# NEVER: Add defensive checks everywhere
def process(self, data):
    if not data:  # 🚫 Unnecessary
        return None
    if not hasattr(data, 'content'):  # 🚫 Let it fail
        return None
    # Just process it!
```

#### Manual Configuration
```python
# NEVER: Create configs in code
config = {  # 🚫 Use YAML files
    "model": "gpt-4",
    "temperature": 0.7
}
```

## Debugging Discipline

### When You Hit Errors

1. **DON'T**: Immediately try to fix
2. **DO**: Understand why it's happening

```bash
# Good debugging flow
1. Reproduce reliably
2. Find minimal test case  
3. Trace execution path
4. Identify divergence point
5. Fix at appropriate level
```

### Validation Error Strategy
```
1. Error occurs: "extra fields not permitted"
2. DON'T: Change extra="forbid" to "allow"
3. DO: 
   - Trace where extra fields come from
   - Find proper transformation point
   - Add field validator or factory method
   - Test the transformation
```

## Testing Guidelines

### Test Categories
- **Unit Tests**: Single function/method
- **Integration Tests**: Component interaction
- **Flow Tests**: End-to-end scenarios

### Test Requirements
```python
# Every bug fix needs:
def test_bug_reproduction():
    """Test that reproduces issue #123."""
    # Should fail before fix
    # Should pass after fix

def test_bug_fix_no_regression():
    """Ensure fix doesn't break existing behavior."""
    # Should pass before and after
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_agent.py::test_specific

# Run with coverage
uv run pytest --cov=buttermilk

# Run in parallel
uv run pytest -n auto
```

## Development Environment

### Essential Setup
```bash
# Install dependencies
uv install

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
export GOOGLE_CLOUD_PROJECT=your-project
```

### Environment Variables
```bash
# Required
export GOOGLE_CLOUD_PROJECT=xxx
export OPENAI_API_KEY=xxx
export ANTHROPIC_API_KEY=xxx

# Optional
export BUTTERMILK_LOG_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
```

### VS Code Settings
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

## Performance Considerations

### Async Best Practices
```python
# Good: Concurrent operations
results = await asyncio.gather(
    agent1.process(data1),
    agent2.process(data2),
    agent3.process(data3)
)

# Bad: Sequential operations
result1 = await agent1.process(data1)
result2 = await agent2.process(data2)
result3 = await agent3.process(data3)
```

### Resource Management
```python
# Good: Proper cleanup
class MyAgent(Agent):
    async def initialize(self, **kwargs):
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        await self.session.close()
```

## Documentation Standards

### Code Documentation
```python
def complex_function(
    data: dict[str, Any],
    *,
    validate: bool = True,
    transform: Callable[[Any], Any] | None = None
) -> ProcessedResult:
    """Process complex data with optional transformation.
    
    This function handles the core data processing pipeline,
    with optional validation and transformation steps.
    
    Args:
        data: Input data dictionary with the following structure:
            {
                "content": str,  # Required
                "metadata": dict,  # Optional
                "config": dict  # Optional
            }
        validate: Whether to validate input data
        transform: Optional transformation function
        
    Returns:
        ProcessedResult containing:
            - transformed_data: The processed data
            - metrics: Processing metrics
            - errors: Any validation errors
            
    Raises:
        ValueError: If validation is enabled and data is invalid
        ProcessingError: If transformation fails
        
    Example:
        >>> result = complex_function(
        ...     {"content": "test"},
        ...     transform=lambda x: x.upper()
        ... )
        >>> print(result.transformed_data)
        "TEST"
    """
```

### Commit Messages
```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Testing
- `chore`: Maintenance

## Quality Checklist

Before committing:
- [ ] Tests pass locally
- [ ] No type errors
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Follows existing patterns
- [ ] GitHub issue updated
- [ ] Commit message clear
- [ ] No sensitive data

## Remember

1. **Quality > Speed**: Better to analyze thoroughly than fix repeatedly
2. **Understand > Guess**: Unknown unknowns are dangerous
3. **Test > Hope**: If it's not tested, it's broken
4. **Document > Remember**: Future you will thank present you
5. **Systematic > Clever**: Boring solutions are maintainable solutions

When in doubt: **STOP and ANALYZE**