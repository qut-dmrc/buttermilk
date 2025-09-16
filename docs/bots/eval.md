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

### Agent Relationship Architecture

#### JUDGE Agents (1:n with SCORERs)
- Multiple JUDGE agents (3-5, different models) evaluate each record independently in zero-shot fashion
- Each JUDGE prediction is evaluated by 3-5 SCORER agents using different models
- **Join Pattern**: `scorer.parent_call_id = judge.call_id`

#### SYNTH Agents (m:n with JUDGEs via session, 1:n with SCORERs)  
- SYNTH agents are full groupchat participants, not simple pipeline steps
- Context includes ALL JUDGE results from the session (not limited to single JUDGE)
- Multiple SYNTH agents (3-5, different models) run per session after all JUDGEs complete
- Each SYNTH prediction is evaluated by 3-5 SCORER agents
- **Join Patterns**: 
  - SYNTH to JUDGEs: `synth.session_id = judge.session_id`
  - SYNTH to SCORERs: `scorer.parent_call_id = synth.call_id`

#### SCORER Evaluation Process
- **Scoring Method**: Each SCORER provides T/F ratings for 1-5 key points from golden answer
- **Non-deterministic**: Number of assessment points varies per scorer evaluation
- **Correctness Calculation**: `correctness = AVG(int(assessment))` per scorer evaluation
- **Multiple Scores**: Each JUDGE/SYNTH prediction gets multiple `correctness` values (one per SCORER)
- **Model Diversity**: Different SCORER models (GPT-4, Claude, Gemini) evaluate same prediction

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

### Key Metrics
1. **Accuracy**: Agreement of prediction (bool) with golden set expected (bool)
2. **Inter-rater Reliability**: Consistency between JUDGE agents
3. **Synthesis Quality**: Improvement from JUDGE to SYNTH
4. **Correctness**: average of boolean LLM-based SCORER assessment on qualitative criteria

## DBT & Analytics Core Principles
1. **Always Handle Deduplication**: Use `DISTINCT call_id` or appropriate window functions
2. **Unnest Nested Fields**: JSON fields need proper extraction
3. **Filter Test Runs**: Exclude debug/test runs using config hashes
4. **Aggregate Correctly**: Account for multiple SCORER evaluations per prediction
5. **Maintain Traceability**: Preserve flow_id and call_id for audit trails
6. Separation of Concerns: DBT handles transformations (calculation engine), dashboards handle display only
7. Single Source of Truth: Each calculation defined once in DBT, parameterized for variations
8. Fail Fast: Tests block bad data from reaching analysis

These principles ensure rigorous, reproducible research while minimizing maintenance overhead.

#### Data Pipeline Design

- Incremental Layers: Raw → Staging → Marts, each layer adds validation
- Modular Transformations: Use Jinja templating for flexible GROUP BY combinations
- Version Everything: SQL transformations in git for reproducibility

#### Validation Strategy

- Test Assumptions Explicitly: Every data assumption becomes an automated test
- Two-Level Testing: Schema tests (structure) + Data tests (business logic)
- Pre-Analysis Gates: CI/CD blocks analysis if tests fail

#### DRY Implementation

- Parameterize, Don't Duplicate: One flexible model > many static views
- Metrics Layer: Define calculations once, query with different dimensions
- Shared Base Queries: Dashboard variations call common validated functions

#### Critical Data Analysis Implications

##### For Performance Comparison (JUDGE vs SYNTH):
- **Cannot use parent_call_id** to link SYNTH to specific JUDGE - SYNTH sees all JUDGEs
- **Must use session_id** to group JUDGE and SYNTH predictions from same session/record
- **Multiple agents per session**: Need aggregation strategy for multiple JUDGEs and SYNTHs per session
- **Session-level analysis**: Comparisons must be within-session, not cross-session

##### For SCORER Analysis:
- **Multiple correctness values** per prediction (one per SCORER model)
- **Deduplication critical**: Use `DISTINCT call_id` when aggregating predictions
- **SCORER diversity**: Different models (GPT-4, Claude, Gemini) provide different correctness scores
- **Non-deterministic assessments**: Number of T/F ratings varies per scorer evaluation

##### For Synth Lift Analysis:
- **Session-based comparison**: Compare aggregated SYNTH vs aggregated JUDGE performance within sessions
- **Aggregation strategy required**: How to handle multiple JUDGEs and SYNTHs per session (AVG, MAX, etc.)
- **SCORER consistency**: Ensure same SCORER models evaluate both JUDGE and SYNTH for fair comparison
- **Temporal ordering**: SYNTH always runs after all JUDGEs complete, provides synthesis not iteration

#### Operational Excellence

- Automated Refresh: Trigger on events (new data) or schedule
- Audit Trail: Test results stored and queryable
- Loud Failures: Errors surface immediately in CI and monitoring

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
- **Streamlit**: ./examples/tja/
- **Looker Studio**: https://lookerstudio.google.com/u/0/reporting/ce580cdd-794a-455d-a4c3-891b6f6b3305/page/p_cgf1sgewvd
- **Google Sheets**: https://docs.google.com/spreadsheets/d/1N98c28IZE9xjvAUp2vLE8cIR6E7qoq7aFUyi0dLrSMU
