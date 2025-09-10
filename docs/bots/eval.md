# Automod Evaluation System Documentation

This document serves as a comprehensive guide for LLM agents assisting with evaluation, analysis, and DBT query creation for the Automod project.

## Project Overview

**Automod** is a high-impact public demonstration of the Buttermilk research platform, designed as a practical accountability tool for evaluating AI-generated content against complex guidelines. The system focuses on rigorous evaluation of LLM capabilities in applying nuanced content moderation rules, particularly around sensitive topics like trans journalism guidelines (TJA) and GLAAD media standards.

## Core Goals

1. **Academic Rigor**: Enable humanities, arts, and social sciences (HASS) scholars to conduct reproducible, traceable computational research
2. **Content Moderation Evaluation**: Systematically assess how well different LLMs can apply complex editorial guidelines to text
3. **Accountability**: Provide transparent, auditable evaluation of AI systems' performance on sensitive content
4. **Accessibility**: Allow non-programmers to run experiments and analyze results without technical expertise

## System Architecture

### Data Flow Pipeline

```
Input (Golden Set) → JUDGE Agents → SYNTH Agent → SCORER Agents → BigQuery → Analysis
```

1. **Golden Set**: Hand-coded examples with detailed, criterion-referenced assessments
2. **JUDGE Round**: Multiple LLM agents (Gemini, Claude, GPT-4) make zero-shot decisions on the same text
3. **SYNTH Round**: Meta-analyst agent reviews all JUDGE answers to produce robust final answer
4. **SCORER Round**: Evaluates every JUDGE and SYNTH answer against golden set using multiple LLMs
5. **Storage**: Results stored as `AgentTrace` records in BigQuery
6. **Analysis**: Looker Studio dashboards and Google Sheets for visualization

## Data Structure

### Primary BigQuery Table
- **Location**: `prosocial-443205.testing.flows`
- **Record Type**: `AgentTrace` objects (Pydantic models)
- **Key Fields**:
  - `call_id`: Unique identifier for each prediction (CRITICAL for deduplication)
  - `flow_id`: Identifies the experimental run
  - `agent_name`: Which agent produced this trace (JUDGE, SYNTH, SCORER, etc.)
  - `model`: LLM used (sonnet, gemini-2.5-pro, gpt-5-mini, etc.)
  - `config_hash`: Hash of configuration for experiment tracking
  - `template_hash`: Hash of system prompt template for A/B testing
  - `record_hash`: Hash of input record from golden set
  - `timestamp`: When the prediction was made
  - `output`: The actual prediction/evaluation result
  - `metadata`: Additional context (criteria used, prompt strategy, etc.)

### Data Normalization Challenges
- **CRITICAL**: Raw data is highly normalized with nested structures
- **Multiple Rows Per Prediction**: Due to joins with SCORER evaluations, a single JUDGE/SYNTH prediction appears in multiple rows
- **Deduplication Required**: Always use `DISTINCT call_id` when aggregating
- **Nested Fields**: Qualitative and quantitative ratings are nested JSON that need unnesting

## Evaluation Criteria Sets

### Available Criteria
1. **TJA (Trans Journalists Association)**: Complex guidelines for trans-inclusive journalism
2. **GLAAD**: Media accountability standards for LGBTQ+ representation
3. **Australian**: Regional content moderation guidelines
4. **Simplified**: Streamlined rules for baseline testing

### Experiment Parameters
Each experimental run is uniquely defined by:
- **Model**: The LLM being evaluated
- **Criteria**: Which guideline set to apply
- **Template**: System prompt configuration (hashed for A/B testing)
- **Strategy**: JUDGE (single-shot) vs SYNTH (consensus)
- **Record**: Specific example from golden set
- **Iteration**: Run number (minimum 10 for stochastic testing)

## DBT Query Development Guidelines

### Core Principles
1. **Always Handle Deduplication**: Use `DISTINCT call_id` or appropriate window functions
2. **Unnest Nested Fields**: JSON fields need proper extraction
3. **Filter Test Runs**: Exclude debug/test runs using config hashes
4. **Aggregate Correctly**: Account for multiple SCORER evaluations per prediction
5. **Maintain Traceability**: Preserve flow_id and call_id for audit trails

### Common Query Patterns

#### 1. Basic Prediction Accuracy
```sql
-- Get unique predictions with their scores
WITH unique_predictions AS (
  SELECT DISTINCT
    call_id,
    flow_id,
    agent_name,
    model,
    output,
    timestamp
  FROM flows
  WHERE agent_name IN ('JUDGE', 'SYNTH')
)
```

#### 2. Scorer Aggregation
```sql
-- Aggregate scorer evaluations per prediction
WITH scorer_summary AS (
  SELECT
    parent_call_id,
    AVG(CAST(JSON_EXTRACT_SCALAR(output, '$.score') AS FLOAT64)) as avg_score,
    COUNT(DISTINCT scorer_model) as num_scorers
  FROM flows
  WHERE agent_name = 'SCORER'
  GROUP BY parent_call_id
)
```

#### 3. Model Performance Comparison
```sql
-- Compare model performance by criteria
SELECT
  model,
  criteria,
  AVG(accuracy) as avg_accuracy,
  STDDEV(accuracy) as accuracy_stddev,
  COUNT(*) as num_predictions
FROM aggregated_results
GROUP BY model, criteria
```

#### 4. Stochastic Stability Analysis
```sql
-- Check consistency across multiple runs
WITH run_variance AS (
  SELECT
    model,
    record_hash,
    VAR_POP(score) as score_variance,
    COUNT(*) as num_runs
  FROM predictions
  GROUP BY model, record_hash
  HAVING COUNT(*) >= 10
)
```

## Key Validation Requirements

### Essential Checks
1. **Completeness**: Ensure all expected agents (FETCH, JUDGE, SYNTH, SCORER, DIFF) have outputs
2. **Consistency**: Verify predictions are stable across multiple runs (variance < threshold)
3. **Coverage**: Check all golden set examples have been evaluated
4. **Model Comparison**: Assess relative performance across different LLMs
5. **Criteria Effectiveness**: Evaluate how well different guideline sets perform

### Data Quality Indicators
- **Missing Scores**: Flag predictions without SCORER evaluations
- **Outlier Detection**: Identify predictions with unusual variance
- **Timestamp Gaps**: Detect incomplete or interrupted flows
- **Hash Mismatches**: Ensure experiment configuration consistency

## Analysis Tools

### Primary Dashboards
- **Looker Studio**: https://lookerstudio.google.com/u/0/reporting/ce580cdd-794a-455d-a4c3-891b6f6b3305/page/p_cgf1sgewvd
- **Google Sheets**: https://docs.google.com/spreadsheets/d/1N98c28IZE9xjvAUp2vLE8cIR6E7qoq7aFUyi0dLrSMU

### Key Metrics
1. **Accuracy**: Agreement with golden set answers
2. **Inter-rater Reliability**: Consistency between JUDGE agents
3. **Synthesis Quality**: Improvement from JUDGE to SYNTH
4. **Model Rankings**: Comparative performance across LLMs
5. **Criteria Difficulty**: Which guidelines are hardest to apply

## DBT Model Structure

### Recommended DBT Models

#### Staging Layer (`stg_`)
- `stg_flows.sql`: Clean and type-cast raw flow data
- `stg_predictions.sql`: Extract JUDGE/SYNTH predictions
- `stg_scores.sql`: Extract SCORER evaluations

#### Intermediate Layer (`int_`)
- `int_predictions.sql`: Deduplicated predictions with basic metrics
- `int_scores_aggregated.sql`: Aggregated scores per prediction

#### Marts Layer
- `judge_scores.sql`: Final analysis-ready JUDGE performance data
- `synth_quality.sql`: SYNTH improvement metrics
- `model_comparison.sql`: Cross-model performance analysis
- `criteria_analysis.sql`: Guideline-specific insights

### DBT Best Practices
1. **Use CTEs**: Structure queries with clear Common Table Expressions
2. **Document Models**: Add schema.yml with descriptions and tests
3. **Test Assumptions**: Validate unique keys, not-null constraints
4. **Version Control**: Track all DBT changes in git
5. **Incremental Models**: Consider for large-scale production runs

## Common Pitfalls to Avoid

1. **Forgetting Deduplication**: Multiple rows per prediction due to SCORER joins
2. **Ignoring Test Runs**: Include proper filters for production analysis
3. **Mishandling JSON**: Nested fields need proper extraction functions
4. **Incorrect Aggregation**: Not accounting for hierarchical data structure
5. **Missing Null Checks**: Handle cases where agents fail or timeout

## Validation Workflow

### Step 1: Data Completeness Check
Verify all expected components of a flow are present

### Step 2: Score Validation
Ensure SCORER agents are properly evaluating predictions

### Step 3: Statistical Analysis
Check for sufficient runs and acceptable variance

### Step 4: Performance Metrics
Calculate accuracy, consistency, and improvement metrics

### Step 5: Comparative Analysis
Compare across models, criteria, and experimental conditions

## Query Templates for Common Tasks

### Get Latest Experiment Results
```sql
SELECT * FROM flows 
WHERE flow_id IN (
  SELECT DISTINCT flow_id 
  FROM flows 
  WHERE DATE(timestamp) = CURRENT_DATE()
)
```

### Find Problematic Predictions
```sql
-- Predictions with high scorer disagreement
SELECT 
  call_id,
  MAX(score) - MIN(score) as score_range
FROM scorer_evaluations
GROUP BY call_id
HAVING score_range > 0.5
```

### Model Performance Summary
```sql
-- Overall model performance
SELECT 
  model,
  COUNT(DISTINCT call_id) as total_predictions,
  AVG(accuracy) as avg_accuracy,
  PERCENTILE_CONT(accuracy, 0.5) as median_accuracy
FROM model_predictions
GROUP BY model
ORDER BY avg_accuracy DESC
```

## Next Steps for Evaluation

1. **Expand Golden Set**: Add more diverse, challenging examples
2. **Refine Scoring**: Develop more nuanced SCORER prompts
3. **Add Criteria**: Incorporate additional guideline sets
4. **Improve Stability**: Increase minimum runs for stochastic testing
5. **Automate Analysis**: Build scheduled DBT runs for regular reporting

## Contact & Resources

- **BigQuery Project**: `prosocial-443205`
- **Main Table**: `testing.flows`
- **DBT Models**: `/dbt/models/`
- **Configuration**: `/conf/` directory (YAML files)
- **Documentation**: `/docs/bots/` for additional context

Remember: The goal is rigorous, reproducible evaluation that helps HASS scholars understand and improve AI content moderation capabilities while maintaining academic integrity and transparency.