# Automod Evaluation System Documentation

This document serves as a comprehensive guide for LLM agents assisting with evaluation, analysis, and DBT query creation for the Automod project.

**Automod** provides an LLM groupchat pattern for the automated application of criteria to content. It uses the Buttermilk research platform to provide a practical tool for the automatic evaluation of content against complex guidelines. The system focuses on rigorous evaluation of LLM capabilities in applying nuanced content moderation rules.

Several distinct research projects utilise the Automod academic research flow environment, including:

* **TJA**, the Trans Journalism ethical guidelines project, using 28 hand-coded news articles on trans issues (includes satire).
*   **ChatGPT vs Oversight Board:** Facebook Oversight Board hate speech decisions, evaluated against FB rules and our custom prompts.
*   **GBV:** News articles on sexual violence, evaluated against professional journalistic ethics.
*   **Toxicity:** 20 examples from Thiago's "Silencing Drag Queens" paper, comparing commercial toxicity models to our custom prompts.

## Core Goals

1. **Academic Rigor**: Enable humanities, arts, and social sciences (HASS) scholars to conduct reproducible, traceable computational research
2. **Content Moderation Evaluation**: Systematically assess how well different LLMs can apply complex editorial guidelines to text
3. **Accountability**: Provide transparent, auditable evaluation of AI systems' performance on sensitive content
4. **Accessibility**: Allow non-programmers to run experiments and analyze results without technical expertise

## Core workflow

Automod projects use different combinations of prediction and validation processes to ensure quality and consistency.

### Trans Flow Pipeline

```
Input (Golden Set) → JUDGE Agents → SYNTH Agent → SCORER Agents → BigQuery → Analysis
```

1. **Golden Set**: Hand-coded examples with detailed, criterion-referenced assessments
2. **JUDGE Round**: Multiple LLM agents (Gemini, Claude, GPT-4) make zero-shot decisions on the same text
3. **SYNTH Round**: Meta-analyst agent reviews all JUDGE answers to produce robust final answer
4. **SCORER Round**: Evaluates every JUDGE and SYNTH answer against golden set using multiple LLMs
5. **Storage**: Results stored as `AgentTrace` records in BigQuery
6. **Analysis**: Looker Studio dashboards and Google Sheets for visualization

#### Applicable criteria

Each flow run assesses a record according to one set of criteria:
1. **TJA (Trans Journalists Association)**: Complex guidelines for trans-inclusive journalism
2. **GLAAD**: Media accountability standards for LGBTQ+ representation
3. **Australian**: Regional content moderation guidelines
4. **Simplified**: Streamlined rules for baseline testing


## Architecture

- **Input records:** Project datasets are stored in **Google Cloud Storage (GCS)**.
- **Results:** All evaluation outputs are stored as `AgentTrace` objects in the BigQuery table `prosocial-443205.testing.flows`. Traces are additionally sent to **Weights and Biases**.
- **Stochastic Testing:** Every prediction should be run at least 10 times to ensure results are stable.
- **Test Configuration:** An experimental run is defined by: Language Model, System Prompt (hashed), Evaluation Criteria, Prompting Strategy (`JUDGE`/`SYNTH`), and the Example from the golden set.
-   **Experiment Tracking**: To manage A/B testing, we use hashes of the `config`, `template`, and `record` for each run. This allows for precise aggregation of results from identical setups and helps filter out test/debug runs.

### Data Normalization Challenges
- **CRITICAL**: Raw data is highly normalized with nested structures
- **Multiple Rows Per Prediction**: Due to joins with SCORER evaluations, a single JUDGE/SYNTH prediction appears in multiple rows
- **Deduplication Required**: Always use `DISTINCT call_id` when aggregating
- **Nested Fields**: Qualitative and quantitative ratings are nested JSON that need unnesting

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

