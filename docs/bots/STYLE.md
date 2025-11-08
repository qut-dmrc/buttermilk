# Buttermilk Style Guide

**Load framework STYLE.md for generic code conventions** (`@$ACADEMICOPS/core/STYLE.md`).

This file contains Buttermilk-specific style requirements.

## Mission-Driven Conventions

**Buttermilk serves HASS scholars** - code must be:

- **Understandable**: Clear naming, well-documented
- **Traceable**: Structured logging, observable execution
- **Reproducible**: Deterministic, versioned, testable

## Naming Conventions (Buttermilk-Specific)

**Agents**:

- `FetchAgent`, `JudgeAgent`, `SynthAgent` (PascalCase with "Agent" suffix)
- Located in `buttermilk/flows/agents/`

**Flows**:

- `trans` (transgender representation flow)
- Configuration in `conf/flows/[flow_name].yaml`
- Implementation in `buttermilk/flows/[flow_name]_flow.py`

**Data Sources**:

- `ZoteroSource`, `TMDBSource` (PascalCase with "Source" suffix)
- Located in `buttermilk/data/sources/`

**Processors**:

- `SemanticSplitter`, `EmbeddingGenerator` (descriptive PascalCase)
- Located in `buttermilk/processors/`

**Storage**:

- `ChromaDBEmbeddings`, `BigQueryStorage` (tech + purpose)
- Located in `buttermilk/data/`

## Configuration Patterns

**Hydra Composable Configs** (`conf/`):

```yaml
# conf/flows/trans.yaml
defaults:
  - /criteria: tja
  - /sources: zotero

flow_name: trans
record_id: ${record_id} # Runtime override
agents:
  - fetch
  - judge
  - synth
```

**Environment Variables** (`.env`):

```bash
# NEVER commit real values - use .env.example for documentation
VERTEX_AI_PROJECT=your-project-id
VERTEX_AI_LOCATION=us-central1
ZOTERO_API_KEY=your-api-key
```

## Logging Conventions

**Structured Logging** (required):

```python
from buttermilk._core.log import logger

# ✅ Good - structured with context
logger.info(
    "agent_started",
    agent_name="FetchAgent",
    record_id=record.record_id,
    flow_name=flow_name,
)

# ✅ Good - error with exception
logger.error("agent_failed", agent_name="JudgeAgent", error=str(e), exc_info=True)

# ❌ Bad - unstructured string
logger.info(f"FetchAgent started for {record.record_id}")
```

**Log Levels**:

- `ERROR`: Agent failure, API errors, critical problems
- `WARNING`: Unexpected conditions, fallbacks used
- `INFO`: Agent lifecycle, significant events (start/complete)
- `DEBUG`: Detailed agent internals, prompt construction

## Documentation Standards

**Agent Docstrings**:

```python
class JudgeAgent:
    """Evaluate content against criteria using LLM judgments.

    The judge agent analyzes content (articles, documents, media)
    and scores them against specified criteria (e.g., GLAAD guidelines,
    TJA stylebook). Returns structured judgments with reasoning.

    Attributes:
        llm_client: Client for LLM API calls (Vertex AI)
        criteria: Evaluation criteria configuration
        temperature: LLM temperature (0.0 for consistency)
    """
```

**Flow Documentation**:

```python
async def run_trans_flow(record_id: str, criteria: str) -> FlowResult:
    """Run transgender representation analysis flow.

    This flow evaluates media content for transgender representation
    quality using multi-agent orchestration:
    1. Fetch: Retrieve content from Zotero
    2. Judge: Evaluate against TJA/GLAAD criteria
    3. Synth: Synthesize findings into report
    4. Scorer: Calculate aggregate scores
    5. Diff: Compare against baselines

    Args:
        record_id: Unique identifier for content to analyze
        criteria: Evaluation criteria ("tja", "glaad")

    Returns:
        FlowResult with agent outputs, scores, and synthesized report

    Raises:
        FlowExecutionError: If any agent fails
        ValidationError: If record_id or criteria invalid
    """
```

## Reproducibility Patterns

**Deterministic LLM Calls**:

```python
# ✅ Good - deterministic for reproducibility
llm_client.generate(
    prompt=prompt,
    temperature=0.0,  # Deterministic
    seed=42,  # Reproducible
    max_tokens=1000,
)

# ❌ Bad - non-reproducible
llm_client.generate(
    prompt=prompt,
    temperature=0.7,  # Stochastic
)
```

**Version Tracking**:

```python
# Include model versions in results
result = {
    "judgment": judgment,
    "model": "gemini-1.5-pro-002",
    "criteria_version": "tja-2024-01",
    "timestamp": datetime.utcnow().isoformat(),
}
```

## Academic Rigor

**Citation and Attribution**:

```python
# Document criteria sources
CRITERIA_SOURCES = {
    "tja": "Trans Journalists Association Media Reference Guide, 2023",
    "glaad": "GLAAD Media Reference Guide, 11th Edition",
}
```

**Transparent Evaluation**:

```python
# Include reasoning with judgments
judgment = {
    "score": 4.5,
    "reasoning": "Article uses correct pronouns consistently...",
    "evidence": ["Quote 1", "Quote 2"],
    "criteria_applied": ["correct_pronouns", "avoid_deadnaming"],
}
```

## Error Messages (Domain-Specific)

**Clear, Actionable Errors**:

```python
# ✅ Good - actionable for HASS researchers
raise ValueError(
    f"Record ID '{record_id}' not found in Zotero library. "
    f"Check library_id in conf/sources/zotero.yaml and verify "
    f"record exists in your Zotero collection."
)

# ❌ Bad - technical jargon
raise ValueError(f"Record {record_id} not in DB")
```

## Code Organization (Buttermilk-Specific)

**Directory Structure**:

```
buttermilk/
  ├── _core/           # Infrastructure (execution_context, config, llms)
  ├── flows/           # Multi-agent orchestration
  │   └── agents/      # Flow-specific agents
  ├── data/            # Storage and retrieval
  │   └── sources/     # External data sources
  ├── processors/      # Data transformation
  └── runner/          # CLI and API entrypoints

conf/                  # Hydra configuration
  ├── flows/           # Flow definitions
  ├── sources/         # Data source configs
  └── criteria/        # Evaluation criteria

tests/
  ├── unit/            # Component tests
  ├── integration/     # Multi-component tests
  └── endtoend/        # Full flow tests (MANDATORY)
```

See framework STYLE.md for generic Python conventions and formatting.
