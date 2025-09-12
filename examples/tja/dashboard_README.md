# TJA Template Analysis Dashboard

## Overview

This Streamlit dashboard provides comprehensive analysis of Trans Journalism Association (TJA) template A/B testing experiments. It integrates with the Buttermilk DBT data pipeline to deliver real-time insights into model performance, data quality, and experimental integrity.

## Key Features

### 📊 Template Performance Analysis
- Overall accuracy comparison across templates
- Performance breakdown by model and criteria
- Statistical significance testing and lift measurement
- Multi-dimensional radar charts and heatmaps

### 🔍 Data Quality Monitoring
- **Score Quality Tracking**: Monitors valid, duplicate-resolved, and missing scores
- **Data Sufficiency Alerts**: Warns when insufficient data threatens analysis validity
- **Experiment Integrity**: Validates A/B test structure and balance
- **Pipeline Completeness**: Tracks FETCH→JUDGE→SYNTH→SCORER→DIFF flow completion

### ⚡ Real-time Integration with DBT
- **Graceful Failure Handling**: Excludes incomplete experimental runs without breaking analysis
- **Incremental Data Awareness**: Shows data freshness and lag indicators
- **Configuration Alignment**: Uses DBT project variables for consistent thresholds
- **Quality Test Integration**: Displays results from automated DBT data quality tests

## Dashboard Sections

### 1. Overall Template Performance
- Bar charts with error bars showing accuracy ± standard deviation
- Total experiment counts and prediction volumes
- Template comparison with statistical confidence indicators

### 2. Data Completeness & Gap Detection
- Coverage matrix heatmap across model-role-criteria combinations
- Data sufficiency indicators with configurable thresholds
- Visual alerts for insufficient experimental runs

### 3. Template Comparison Analysis
- Performance breakdown by language model
- Criteria-specific effectiveness measurements
- Side-by-side template comparisons

### 4. Lift Measurement & Statistical Validation
- Scatter plots showing absolute vs. relative lift
- Forest plots for statistically significant improvements
- Effect size calculations and confidence intervals

### 5. Multi-Dimensional Analysis
- Performance heatmaps by template
- Radar charts comparing templates across dimensions
- Balanced comparison requiring multiple template variants

### 6. Data Quality Dashboard
- **Data Sufficiency Tab**: Coverage rates and run count monitoring
- **Experiment Integrity Tab**: A/B test balance and structure validation
- **Pipeline Completeness Tab**: Flow completion rates and common issues
- **Score Quality Distribution**: Valid/duplicate/missing score breakdown

### 7. Actionable Insights
- Top-performing configurations ranked by accuracy
- Improvement opportunities with statistical backing
- Quality metrics and data freshness indicators

## Configuration & Filters

### Sidebar Controls
- **Criteria Selection**: Filter by evaluation guidelines (TJA, GLAAD, etc.)
- **Model Selection**: Focus on specific language models
- **Agent Role Selection**: Filter by JUDGE, SYNTHESISER, SCORERS
- **Data Sufficiency**: Show only sufficient/insufficient data
- **Score Quality**: Include/exclude different quality categories
- **Data Freshness**: Filter by hours since experimental run

### Automatic Configuration
- Thresholds automatically loaded from DBT project variables
- Data quality standards aligned with backend validation
- Graceful fallbacks when DBT tests unavailable

## Technical Integration

### DBT Integration
```python
# Loads DBT test results and metrics
config = DashboardConfig(bm)
min_runs = config.min_stochastic_runs  # From DBT vars
coverage_threshold = config.min_coverage_threshold
```

### Data Quality Monitoring
```sql
-- Queries DBT test results directly
SELECT * FROM assert_data_sufficiency WHERE insufficiency_reason IS NOT NULL
SELECT * FROM assert_experiment_integrity  
SELECT * FROM assert_complete_pipeline
```

### Performance Optimization
- **Cached Data Loading**: Streamlit caching for expensive BigQuery operations
- **Incremental Awareness**: Queries only include fresh data from incremental models
- **Pre-aggregated Marts**: Uses DBT marts instead of raw calculations
- **Lazy Loading**: Heavy visualizations load on-demand

## Usage Guide

### Starting the Dashboard
```bash
cd examples/tja/
streamlit run streamlit_dashboard.py
```

### Interpreting Results

#### Green Indicators ✅
- All data quality checks passed
- Sufficient data for statistical analysis
- Complete pipeline execution
- Valid score coverage above threshold

#### Yellow Warnings ⚠️
- Data quality issues detected but analysis possible
- Some incomplete experimental runs
- Moderate statistical confidence
- Score coverage below optimal but above minimum

#### Red Alerts 🚨
- Insufficient data for reliable analysis
- Major pipeline failures
- Statistical significance too low
- Critical experimental integrity issues

### Best Practices

1. **Always Check Data Quality First**: Review the Data Quality Dashboard before interpreting results
2. **Filter by Score Quality**: Use "valid" and "duplicate_resolved" scores for analysis
3. **Respect Data Sufficiency**: Don't draw conclusions from insufficient data warnings
4. **Monitor Data Freshness**: Check timestamps to ensure you're viewing current results
5. **Validate Statistical Significance**: Only trust results with adequate confidence levels

## Troubleshooting

### Common Issues

**"Could not load DBT quality data"**
- DBT models may not be compiled/run
- Check BigQuery permissions
- Verify table names match deployed models

**"No data available for selected filters"**
- Filters too restrictive
- Data freshness threshold too aggressive
- Check if recent experimental runs completed

**Quality tests showing many issues**
- Normal for experimental research
- Focus on overall trends, not individual failures
- Ensure minimum run counts before analysis

### Support

For technical issues:
1. Check DBT logs: `dbt test --severity warn`
2. Verify data pipeline: Review pipeline completeness tab
3. Validate configuration: Ensure DBT variables are set correctly

For research questions:
- Consult the TJA guidelines documentation
- Review experimental design assumptions
- Consider statistical power analysis for minimum sample sizes