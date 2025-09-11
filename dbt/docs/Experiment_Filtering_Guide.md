# Experiment Filtering Guide: Template Hash-Based Analysis

## Understanding the Data Structure

After analyzing your data, I now understand that:

- **Experiments are defined by template hashes**, not session_id
- **Session_id** represents individual execution runs that can happen at any time on any machine
- **Template hashes** represent different prompt versions that you want to compare
- **Multiple experiments** can run concurrently using different template combinations

## Current Template Structure

Your TJA data contains these template hashes:

### JUDGE Templates:
- **Primary (8c95...)**: Used 715 times across multiple criteria (most frequent)
- **Secondary (a559...)**: Used 74 times across multiple criteria (experimental)

### SYNTHESISER Templates:
- **Single template (8972...)**: Used 399 times (no comparison available)

## How to Filter Experiments

### Option 1: Dashboard Filtering (Recommended)

**Best for:** Interactive analysis, quick comparisons, multiple users

**Implementation:**
1. **In Looker Studio:**
   - Add filter controls for `template_hash` or `template_short_hash`
   - Add filter for `experiment_group` (e.g., "JUDGE - tja")
   - Users can select which template hashes to compare

2. **In Google Sheets:**
   - Use filter buttons on `template_hash` columns
   - Create dropdown filters for `experiment_group`

**Example Dashboard Filters:**
```sql
-- Filter for TJA Judge experiment (comparing 2 templates)
WHERE experiment_group = 'JUDGE - tja' 
  AND template_hash IN (
    'sha256:8c9564342aee1e55089369885aa63ce7486c30c8075a561f3f329796b487dd97',
    'sha256:a559f85293f1944a9de6fb8cd72e45c3f6dfee3336015033d0a82e04f7e73ba5'
  )

-- Filter for GLAAD Judge experiment
WHERE experiment_group = 'JUDGE - glaad'
```

### Option 2: DBT Variables (For Automated Reports)

**Best for:** Automated reporting, CI/CD pipelines, scheduled analysis

**Implementation:**
Add to your `dbt_project.yml`:
```yaml
vars:
  experiment_templates:
    - 'sha256:8c9564342aee1e55089369885aa63ce7486c30c8075a561f3f329796b487dd97'
    - 'sha256:a559f85293f1944a9de6fb8cd72e45c3f6dfee3336015033d0a82e04f7e73ba5'
  experiment_criteria: 'tja'
  experiment_agent_role: 'JUDGE'
```

Then modify your models:
```sql
WHERE template_hash IN {{ var('experiment_templates') }}
  AND judge_criteria = '{{ var('experiment_criteria') }}'
  AND agent_role = '{{ var('experiment_agent_role') }}'
```

### Option 3: Experiment-Specific Models

**Best for:** Complex, ongoing experiments with specific requirements

**Implementation:**
Create experiment-specific models like:
- `tja_judge_comparison.sql`
- `glaad_judge_comparison.sql`

## Current Experiment Options

Based on your data, here are the meaningful experiments you can run:

### 1. TJA Judge Templates Comparison
```sql
SELECT template_short_hash, template_label, avg_accuracy, total_predictions
FROM template_performance_comparison 
WHERE experiment_group = 'JUDGE - tja'
  AND data_sufficiency = 'SUFFICIENT'
ORDER BY total_predictions DESC
```

**Result Preview:**
- Template A (8c95...): 96 predictions, 67% accuracy
- Template B (a559...): 4 predictions, 42% accuracy ⚠️ Insufficient data

### 2. GLAAD Judge Templates Comparison
```sql
WHERE experiment_group = 'JUDGE - glaad'
```

### 3. Cross-Criteria Analysis (Same Templates, Different Guidelines)
```sql
SELECT experiment_group, template_short_hash, avg_accuracy
FROM template_performance_comparison
WHERE template_hash = 'sha256:8c9564342aee1e55089369885aa63ce7486c30c8075a561f3f329796b487dd97'
  AND data_sufficiency = 'SUFFICIENT'
ORDER BY experiment_group
```

## Dashboard Configuration Recommendations

### Essential Filters to Add:

1. **Primary Filters:**
   - `experiment_group` dropdown (e.g., "JUDGE - tja", "JUDGE - glaad")
   - `template_short_hash` multi-select
   - `data_sufficiency` toggle (show only SUFFICIENT data)

2. **Secondary Filters:**
   - `agent_role` (JUDGE vs SYNTHESISER)
   - `criteria` (individual criteria selection)
   - `model` (if comparing across different LLMs)

3. **Date Filters:**
   - `first_prediction` / `last_prediction` date range
   - For temporal analysis of template performance

### Sample Filter Combinations:

**Experiment 1: TJA Judge A/B Test**
- experiment_group = "JUDGE - tja"
- template_short_hash IN ["8c9564342aee", "a559f85293f1"]
- data_sufficiency = "SUFFICIENT"

**Experiment 2: Cross-Criteria Template Performance**
- template_short_hash = "8c9564342aee" 
- agent_role = "JUDGE"
- criteria IN ["tja", "glaad", "hrc"]

**Experiment 3: Recent vs Historical Performance**
- template_short_hash = "8c9564342aee"
- first_prediction >= "2025-09-01" (for recent data)

## Quick Start Queries

### 1. List All Available Experiments:
```sql
SELECT 
  experiment_group,
  COUNT(DISTINCT template_hash) as template_count,
  SUM(total_predictions) as total_tests,
  MIN(first_prediction) as earliest_test,
  MAX(last_prediction) as latest_test
FROM template_performance_comparison
WHERE data_sufficiency = 'SUFFICIENT'
GROUP BY experiment_group
ORDER BY total_tests DESC
```

### 2. Template Performance Summary:
```sql
SELECT 
  template_short_hash,
  template_label,
  experiment_group,
  avg_accuracy,
  total_predictions,
  data_sufficiency
FROM template_performance_comparison
WHERE experiment_group = 'JUDGE - tja'  -- Change as needed
ORDER BY total_predictions DESC
```

### 3. Data Coverage Analysis:
```sql
SELECT 
  experiment_group,
  completeness_status,
  COUNT(*) as combination_count
FROM int_experiment_completeness
GROUP BY experiment_group, completeness_status
ORDER BY experiment_group, completeness_status
```

## Implementation Priority

1. **Start with Dashboard Filtering** - Most flexible and user-friendly
2. **Add date range filters** - Essential for temporal analysis  
3. **Create experiment presets** - Quick buttons for common comparisons
4. **Add data sufficiency warnings** - Alert users when sample sizes are too small

This approach gives you maximum flexibility to analyze any combination of templates while maintaining the ability to drill down into specific experiments as needed.