# Buttermilk DBT Documentation

## Overview

This DBT project transforms raw Automod evaluation data into analysis-ready datasets for humanities research. It implements graceful failure handling, ensuring incomplete experimental runs don't break analysis while maintaining rigorous data quality standards.

## Architecture

### Data Flow
```
BigQuery Raw Data → Staging → Intermediate → Marts → Analytics
```

**Staging**: Clean and structure raw `AgentTrace` data
**Intermediate**: Deduplicate scores, track completeness  
**Marts**: Analysis-ready tables with incremental updates
**Metrics**: Semantic layer for consistent calculations

### Key Design Principles

1. **Graceful Degradation**: Exclude incomplete data rather than failing
2. **Data Sufficiency Monitoring**: Alert when insufficient data threatens validity
3. **Experimental Integrity**: Validate A/B test structure and coverage
4. **Incremental Processing**: Efficient updates for large datasets
5. **Parameterized Filtering**: Configurable thresholds and cutoff dates

## Model Layers

### Staging Models
- `stg_flows`: Core cleaned data with JSON extraction
- Handles nested JSON fields from raw AgentTrace records
- Applies basic data type conversions and field extraction

### Intermediate Models  
- `int_predictions`: JUDGE and SYNTHESISER predictions
- `int_scores_deduped`: Deduplicated SCORER evaluations with quality flags
- `int_experiment_completeness`: Pipeline completeness tracking

### Marts Models
- `judge_scores`: Primary analysis table joining predictions with scores
- Uses incremental materialization for performance
- Includes score quality tracking and graceful failure handling

## Testing Strategy

### Schema Tests
- Column-level data quality (unique, not_null, accepted_values)
- Range validation for numeric scores
- Referential integrity between models

### Custom Data Tests
- `assert_data_sufficiency`: Monitors minimum data requirements
- `assert_experiment_integrity`: Validates A/B test structure  
- `assert_complete_pipeline`: Checks evaluation pipeline completeness

### Test Philosophy
- Tests **warn** rather than fail builds for data quality issues
- Focus on detecting insufficient data for analysis validity
- Distinguish between normal experimental failures and pipeline problems

## Configuration

### Variables (dbt_project.yml)
```yaml
vars:
  cutoff_date: "2025-05-01"           # Filter old/bad data
  min_stochastic_runs: 10             # Minimum runs for statistical validity
  min_coverage_threshold: 0.8         # Minimum coverage for analysis
  exclude_test_hashes: []             # Test runs to exclude
```

### Materialization Strategy
- **Staging**: Tables (performance for frequent access)
- **Intermediate**: Views (flexibility during development)  
- **Marts**: Incremental tables (efficiency for large datasets)

## Macros

### JSON Processing
- `extract_json_safely`: Safe JSON extraction with fallbacks
- `unnest_assessments`: Structured assessment data extraction

### Experiment Management  
- `get_experiment_hash`: Consistent experiment grouping
- Handles model/criteria/template/record combinations

## Usage Patterns

### Development Workflow
```bash
# Full refresh during development
dbt run --full-refresh

# Incremental updates in production  
dbt run

# Test data quality
dbt test --severity warn

# Generate documentation
dbt docs generate && dbt docs serve
```

### Analytics Integration
- Metrics layer provides semantic definitions for dashboards
- Parameterized models support multiple analysis dimensions
- Incremental updates minimize computation costs

## Monitoring & Alerts

### Data Quality Monitoring
- Score coverage rates by model/criteria
- Pipeline completion percentages
- Experimental balance validation

### Recommended Alerts
- Coverage below 80% for active experiments
- Pipeline failure rate above 10%
- Missing golden set records for key criteria

## Future Enhancements

### Planned Features
- Automated experiment hash exclusion for failed runs
- Dynamic golden set size detection
- Cross-criteria performance comparison models
- Longitudinal trend analysis tables

### Scalability Considerations
- Partition large fact tables by date
- Consider clustering on high-cardinality dimensions
- Implement snapshot models for slowly changing experiment metadata