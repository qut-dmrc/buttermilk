# Looker Studio Dashboard Design for Template A/B Testing

## Dashboard Structure

### Page 1: Executive Summary
**Purpose**: High-level overview for stakeholders
**Data Source**: `multidimensional_analysis`

**Key Visualizations**:
1. **Overall Winner Card**: Large metric showing which template performs better overall
2. **Overall Lift Chart**: Bar chart showing absolute and relative lift
3. **Confidence Indicators**: Scorecard showing statistical confidence levels
4. **Data Quality Summary**: Table showing completeness by dimension

**Filters**: None (shows overall results)

---

### Page 2: Data Quality & Completeness 
**Purpose**: Validate experiment integrity before making conclusions
**Data Source**: `int_experiment_completeness`

**Key Visualizations**:
1. **Completeness Heatmap**: Matrix showing completion status by Model × Criteria × Agent Type
2. **Missing Data Table**: Detailed list of missing combinations with counts
3. **Prediction Volume Chart**: Bar chart showing number of predictions by template/model/criteria
4. **Time Series**: When experiments were run (identify gaps or issues)

**Filters**: Template, Model, Criteria, Agent Type
**Key Metrics**: Prediction count, completeness status, time ranges

---

### Page 3: Template Performance Deep Dive
**Purpose**: Detailed comparison across all dimensions
**Data Source**: `template_performance_comparison`

**Key Visualizations**:
1. **Accuracy Comparison Chart**: Side-by-side bar chart Template A vs B by model/criteria
2. **Lift Analysis Table**: Sortable table with lift calculations and confidence intervals
3. **Distribution Charts**: Box plots showing accuracy distribution for each template
4. **Statistical Significance Indicators**: Traffic light indicators for confidence levels

**Filters**: Model, Criteria, Agent Type (JUDGE vs SYNTH)
**Key Metrics**: Accuracy, lift (absolute & relative), confidence level, sample size

---

### Page 4: Model Analysis
**Purpose**: Understand which models show the biggest template differences
**Data Source**: `multidimensional_analysis` filtered to MODEL dimension

**Key Visualizations**:
1. **Model Performance Matrix**: Heatmap showing accuracy by Model × Template
2. **Model Lift Rankings**: Horizontal bar chart ranking models by template lift
3. **Model Consistency Chart**: Scatter plot showing accuracy vs variance by model
4. **Interaction Effects**: Line chart showing template performance across models

**Filters**: Criteria, Agent Type
**Key Metrics**: Accuracy by model, lift per model, variance indicators

---

### Page 5: Criteria Analysis  
**Purpose**: Understand which criteria sets show template differences
**Data Source**: `multidimensional_analysis` filtered to CRITERIA dimension

**Key Visualizations**:
1. **Criteria Difficulty Chart**: Bar chart showing baseline accuracy by criteria
2. **Template Sensitivity by Criteria**: Grouped bar chart showing lift by criteria
3. **Criteria Interaction Heatmap**: Template performance across criteria × model combinations
4. **Recommendation Matrix**: Table showing recommendations by criteria

**Filters**: Model, Agent Type  
**Key Metrics**: Baseline accuracy, template lift, effect sizes

---

### Page 6: Judge vs Synth Analysis
**Purpose**: Compare single-agent vs consensus performance
**Data Source**: `template_performance_comparison` 

**Key Visualizations**:
1. **Agent Type Comparison**: Side-by-side metrics for JUDGE vs SYNTH
2. **Synthesis Improvement Chart**: Show how SYNTH improves over JUDGE by template
3. **Agent Stability Analysis**: Compare variance between JUDGE and SYNTH
4. **Template Impact on Synthesis**: How templates affect consensus quality

**Filters**: Model, Criteria
**Key Metrics**: JUDGE accuracy, SYNTH accuracy, improvement from synthesis

---

### Page 7: Statistical Deep Dive
**Purpose**: Advanced statistical analysis for researchers
**Data Source**: `template_performance_comparison` + detailed calculations

**Key Visualizations**:
1. **Effect Size Distribution**: Histogram of effect sizes across dimensions
2. **Confidence Intervals Chart**: Error bars showing accuracy ranges
3. **Power Analysis Table**: Sample size adequacy assessment
4. **Correlation Matrix**: Relationships between accuracy metrics

**Filters**: All dimensions
**Key Metrics**: Effect sizes, confidence intervals, p-values (if calculated), power

---

## Dashboard Implementation Guide

### Data Connection Setup
1. Connect Looker Studio to BigQuery project `prosocial-443205`
2. Add these tables as data sources:
   - `testing.int_experiment_completeness`
   - `testing.template_performance_comparison` 
   - `testing.multidimensional_analysis`

### Key Calculated Fields
```sql
-- Template Winner Indicator
CASE 
  WHEN absolute_lift > 0 THEN "Template B Wins"
  WHEN absolute_lift < 0 THEN "Template A Wins" 
  ELSE "Tie"
END

-- Lift Direction Color
CASE
  WHEN absolute_lift > 0.05 THEN "Green"
  WHEN absolute_lift < -0.05 THEN "Red"
  ELSE "Gray"
END

-- Confidence Level Score
CASE statistical_confidence
  WHEN "HIGH_CONFIDENCE" THEN 3
  WHEN "MEDIUM_CONFIDENCE" THEN 2 
  ELSE 1
END
```

### Color Schemes
- **Templates**: Template A (Blue #4285F4), Template B (Orange #FF6D01)
- **Confidence**: High (Green), Medium (Yellow), Low (Red)
- **Lift Direction**: Positive (Green), Negative (Red), Neutral (Gray)

### Interactive Features
1. **Cross-page filtering**: Selections on one page filter others
2. **Drill-down capability**: Click charts to filter to specific combinations
3. **Export functionality**: Allow CSV export of key tables
4. **Mobile responsive**: Ensure dashboard works on tablets

### Alert Configuration
Set up automated alerts for:
- Data freshness (if no new data in 24 hours)
- Large effect sizes (>10% lift) with high confidence
- Missing data patterns that indicate systematic issues

## Validation Checklist

Before presenting results:
1. ✅ **Data Completeness**: All expected template/model/criteria combinations present
2. ✅ **Sample Size**: Minimum 10 predictions per combination (preferably 30+)
3. ✅ **Time Consistency**: Experiments run in similar timeframes
4. ✅ **Scoring Coverage**: All predictions have SCORER evaluations
5. ✅ **Effect Size**: Lift differences are meaningful (>1% for small, >5% for medium, >10% for large)
6. ✅ **Statistical Confidence**: High confidence for key findings

## Usage Notes

**For Stakeholders**: Start with Executive Summary, then dive into specific dimensions
**For Researchers**: Focus on Data Quality page first, then Statistical Deep Dive
**For Engineers**: Template Performance Deep Dive provides actionable insights

**Key Questions the Dashboard Answers**:
1. Which template performs better overall?
2. Are there data gaps that invalidate conclusions?
3. Which models/criteria show the biggest template differences? 
4. How confident can we be in the results?
5. Should we implement Template B based on the evidence?