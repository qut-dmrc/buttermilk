# TJA Template A/B Testing Dashboard Implementation Guide

## Overview

This guide provides specific chart recommendations for analyzing template performance differences in your TJA experiment using the DBT models we've built. The data shows you have **2 prompt templates** tested across **6 different criteria** with **JUDGE** and **SYNTHESISER** roles.

## Available Data Sources

Your DBT models provide these key tables for dashboard creation:

1. **`int_experiment_completeness`** - Data coverage validation (484 complete combinations)
2. **`template_performance_comparison`** - Core A/B testing results (51 performance records)  
3. **`multidimensional_analysis`** - Cross-dimensional lift analysis
4. **`judge_scores`** - Detailed prediction and scoring data

## Chart Specifications by Category

### 1. Data Completeness & Gap Detection Charts

#### A. Coverage Matrix Heatmap
**Data Source:** `int_experiment_completeness`
```sql
SELECT 
  template_label,
  criteria,
  model,
  agent_role,
  actual_predictions,
  completeness_status
FROM int_experiment_completeness
```

**Chart Type:** Heatmap Matrix
- **Rows:** `criteria` (6 values)
- **Columns:** `model` × `agent_role` combinations
- **Color:** `actual_predictions` (intensity = number of runs)
- **Annotations:** Show `completeness_status` ('MISSING', 'INSUFFICIENT', 'COMPLETE')
- **Goal:** Immediately identify missing or insufficient data

#### B. Data Sufficiency Bar Chart  
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  CONCAT(criteria, ' - ', agent_role) as dimension,
  template_label,
  total_predictions,
  data_sufficiency
FROM template_performance_comparison
ORDER BY total_predictions DESC
```

**Chart Type:** Grouped Bar Chart
- **X-axis:** `dimension` (criteria-role combinations)
- **Y-axis:** `total_predictions`
- **Groups:** `template_label` (Template A vs Template B)
- **Color:** `data_sufficiency` ('SUFFICIENT' = green, 'INSUFFICIENT' = red)
- **Threshold Line:** Horizontal line at 10 predictions (minimum for statistical significance)

### 2. Template Comparison Charts

#### A. Overall Template Performance
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  template_label,
  AVG(avg_accuracy) as overall_accuracy,
  STDDEV(avg_accuracy) as accuracy_variance,
  SUM(total_predictions) as total_tests
FROM template_performance_comparison 
WHERE data_sufficiency = 'SUFFICIENT'
GROUP BY template_label
```

**Chart Type:** Side-by-side Bar Chart with Error Bars
- **X-axis:** `template_label` 
- **Y-axis:** `overall_accuracy`
- **Error Bars:** ±1 standard deviation (`accuracy_variance`)
- **Annotations:** Show `total_tests` on each bar

#### B. Template Performance by Model
**Data Source:** `template_performance_comparison`  
```sql
SELECT 
  model,
  template_label,
  AVG(avg_accuracy) as accuracy,
  AVG(total_predictions) as prediction_count
FROM template_performance_comparison
WHERE data_sufficiency = 'SUFFICIENT'
GROUP BY model, template_label
```

**Chart Type:** Grouped Bar Chart
- **X-axis:** `model` (different LLMs)
- **Y-axis:** `accuracy`
- **Groups:** `template_label` (Template A vs Template B)
- **Size/Annotation:** `prediction_count` (sample size)

#### C. Template Performance by Criteria
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  criteria,
  template_label, 
  AVG(avg_accuracy) as accuracy,
  COUNT(*) as model_count
FROM template_performance_comparison
WHERE data_sufficiency = 'SUFFICIENT' 
GROUP BY criteria, template_label
```

**Chart Type:** Grouped Bar Chart
- **X-axis:** `criteria` (6 different guideline sets)
- **Y-axis:** `accuracy`
- **Groups:** `template_label`
- **Annotations:** `model_count` (how many models tested)

### 3. Lift Measurement & Statistical Validation

#### A. Lift Magnitude Scatter Plot
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  criteria,
  model,
  agent_role,
  absolute_lift,
  relative_lift_percent,
  data_sufficiency,
  confidence_level
FROM template_performance_comparison
WHERE absolute_lift IS NOT NULL
```

**Chart Type:** Scatter Plot
- **X-axis:** `absolute_lift` (raw accuracy difference)
- **Y-axis:** `relative_lift_percent` (percentage improvement)
- **Color:** `criteria` (different guideline sets)
- **Shape:** `agent_role` (JUDGE vs SYNTHESISER)
- **Size:** `confidence_level` (HIGH/MEDIUM/LOW)
- **Quadrant Lines:** Vertical line at x=0, horizontal line at y=0

#### B. Statistical Significance Forest Plot
**Data Source:** `multidimensional_analysis`
```sql
SELECT 
  dimension_type,
  dimension_value,
  absolute_lift,
  relative_lift_percent,
  statistical_confidence,
  effect_size,
  performance_direction
FROM multidimensional_analysis
WHERE statistical_confidence IN ('HIGH_CONFIDENCE', 'MEDIUM_CONFIDENCE')
ORDER BY ABS(relative_lift_percent) DESC
```

**Chart Type:** Forest Plot (Horizontal Bar Chart)
- **Y-axis:** `dimension_value` (ordered by lift magnitude)
- **X-axis:** `relative_lift_percent`
- **Color:** `statistical_confidence` (HIGH = dark, MEDIUM = light)
- **Shape:** `effect_size` (LARGE/MEDIUM/SMALL = different shapes)
- **Reference Line:** Vertical line at x=0 (no difference)

### 4. Multi-Dimensional Analysis

#### A. Performance Heatmap Matrix
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  model,
  criteria, 
  template_label,
  avg_accuracy,
  confidence_level
FROM template_performance_comparison
WHERE data_sufficiency = 'SUFFICIENT'
```

**Chart Type:** 3D Heatmap (or Faceted Heatmap)
- **Rows:** `model` 
- **Columns:** `criteria`
- **Facets:** `template_label` (separate heatmap for each template)
- **Color Intensity:** `avg_accuracy`
- **Border/Pattern:** `confidence_level`

#### B. Radar Chart Comparison
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  template_label,
  CONCAT(model, ' - ', criteria) as dimension,
  avg_accuracy
FROM template_performance_comparison
WHERE data_sufficiency = 'SUFFICIENT'
```

**Chart Type:** Radar Chart (or Parallel Coordinates)
- **Axes:** Each `dimension` (model-criteria combination)
- **Lines:** One line per `template_label`
- **Scale:** 0 to 1 on each axis (accuracy scale)

### 5. Actionable Insights Charts

#### A. Top Performing Configurations
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  CONCAT(template_label, ' - ', model, ' - ', criteria) as configuration,
  avg_accuracy,
  total_predictions,
  confidence_level,
  accuracy_rank
FROM template_performance_comparison
WHERE data_sufficiency = 'SUFFICIENT'
ORDER BY avg_accuracy DESC
LIMIT 10
```

**Chart Type:** Horizontal Bar Chart (Top 10)
- **Y-axis:** `configuration` (ranked list)
- **X-axis:** `avg_accuracy`
- **Color:** `confidence_level`
- **Annotations:** `total_predictions` (sample size)

#### B. Biggest Improvement Opportunities  
**Data Source:** `multidimensional_analysis`
```sql
SELECT 
  dimension_value,
  absolute_lift,
  relative_lift_percent,
  recommendation,
  statistical_confidence
FROM multidimensional_analysis
WHERE recommendation LIKE '%RECOMMEND_B%'
  AND statistical_confidence != 'LOW_CONFIDENCE'
ORDER BY ABS(relative_lift_percent) DESC
```

**Chart Type:** Waterfall Chart or Grouped Bar Chart
- **X-axis:** `dimension_value` (ordered by improvement magnitude)
- **Y-axis:** `relative_lift_percent`
- **Color:** `recommendation` (STRONG vs WEAK)
- **Pattern:** `statistical_confidence`

### 6. Data Quality Indicators

#### A. Variance Analysis Box Plot
**Data Source:** `template_performance_comparison`
```sql
SELECT 
  template_label,
  criteria,
  avg_accuracy,
  accuracy_stddev,
  min_accuracy,
  max_accuracy,
  median_accuracy
FROM template_performance_comparison
WHERE data_sufficiency = 'SUFFICIENT'
```

**Chart Type:** Box Plot
- **X-axis:** `template_label` × `criteria` combinations
- **Y-axis:** Accuracy distribution (using min, max, median, stddev)
- **Outlier Detection:** Points beyond ±2 standard deviations

## Implementation in Looker Studio / Google Sheets

### Looker Studio Implementation
1. **Data Connection:** Connect to BigQuery dataset `prosocial-443205.bmdev`
2. **Chart Types Available:** All chart types above are supported
3. **Filtering:** Add filters for `criteria`, `model`, `agent_role`, `data_sufficiency`
4. **Interactivity:** Enable cross-chart filtering and drill-down capabilities

### Google Sheets Implementation  
1. **Data Import:** Use BigQuery connector or QUERY() functions
2. **Chart Creation:** Use built-in chart types (may need to combine multiple charts for complex visualizations)
3. **Conditional Formatting:** Use for heatmaps and data quality indicators
4. **Pivot Tables:** Create pivot tables for interactive analysis

## Key Metrics Summary

Based on your data analysis, track these essential KPIs:

1. **Coverage Completeness:** Percentage of experiment combinations with ≥10 runs
2. **Template Lift:** Overall accuracy improvement of Template B over Template A  
3. **Statistical Confidence:** Percentage of comparisons with high/medium confidence
4. **Model Consistency:** Variance in performance across different models
5. **Criteria Effectiveness:** Which guidelines show the largest template differences

## Quick Start Queries

To get started immediately, here are the three most important queries:

```sql
-- 1. Overall template comparison
SELECT template_label, AVG(avg_accuracy) as avg_accuracy, COUNT(*) as comparisons
FROM template_performance_comparison 
WHERE data_sufficiency = 'SUFFICIENT'
GROUP BY template_label;

-- 2. Data gaps identification  
SELECT criteria, model, agent_role, COUNT(*) as missing_combinations
FROM int_experiment_completeness
WHERE completeness_status = 'MISSING'
GROUP BY criteria, model, agent_role;

-- 3. Biggest improvements
SELECT dimension_value, relative_lift_percent, statistical_confidence
FROM multidimensional_analysis
WHERE recommendation = 'STRONG_RECOMMEND_B'
ORDER BY ABS(relative_lift_percent) DESC;
```

These charts will give you comprehensive visibility into template performance differences, data quality, and statistical significance across all dimensions of your TJA experiment.