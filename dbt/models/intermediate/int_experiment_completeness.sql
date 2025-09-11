{{ config(
    materialized='table',
    description='Data completeness validation for template hash-based experiments'
) }}

-- This model supports filtering by specific template hashes to define experiments
-- Use dashboard filters or WHERE clauses to select which template hashes to compare

WITH base_data AS (
  SELECT DISTINCT
    template_hash,
    template_name,
    agent_model as model,
    judge_criteria as criteria,
    record_id as record_hash,
    agent_role,
    session_id,
    timestamp,
    call_id
  FROM {{ ref('stg_flows') }}
  WHERE agent_role IN ('JUDGE', 'SYNTHESISER')
    AND template_hash IS NOT NULL
    AND record_id IS NOT NULL
    AND judge_criteria IS NOT NULL
),

-- Get actual predictions aggregated by template hash
actual_predictions AS (
  SELECT 
    template_hash,
    template_name,
    model,
    criteria,
    record_hash,
    agent_role,
    COUNT(DISTINCT call_id) as prediction_count,
    COUNT(DISTINCT session_id) as session_count,
    MIN(timestamp) as first_run,
    MAX(timestamp) as last_run,
    ARRAY_AGG(DISTINCT session_id LIMIT 10) as sample_sessions
  FROM base_data
  GROUP BY 1, 2, 3, 4, 5, 6
),

-- Create all possible combinations for the selected template hashes
-- This generates the full experimental matrix
expected_combinations AS (
  SELECT DISTINCT
    template_hash,
    template_name,
    model,
    criteria,
    record_hash,
    agent_role
  FROM base_data
),

-- Identify gaps and completeness
completeness_analysis AS (
  SELECT
    e.template_hash,
    e.template_name,
    e.model,
    e.criteria,
    e.record_hash,
    e.agent_role,
    COALESCE(a.prediction_count, 0) as actual_predictions,
    COALESCE(a.session_count, 0) as session_count,
    a.first_run,
    a.last_run,
    a.sample_sessions,
    CASE 
      WHEN a.prediction_count IS NULL THEN 'MISSING'
      WHEN a.prediction_count < 10 THEN 'INSUFFICIENT' -- Need 10+ for stochastic testing
      ELSE 'COMPLETE'
    END as completeness_status
  FROM expected_combinations e
  LEFT JOIN actual_predictions a USING (template_hash, template_name, model, criteria, record_hash, agent_role)
),

-- Add template comparison metadata
template_metadata AS (
  SELECT 
    template_hash,
    template_name,
    agent_role,
    COUNT(DISTINCT criteria) as criteria_count,
    COUNT(DISTINCT model) as model_count,
    SUM(actual_predictions) as total_predictions,
    MIN(first_run) as template_first_used,
    MAX(last_run) as template_last_used
  FROM completeness_analysis
  GROUP BY 1, 2, 3
)

SELECT
  -- Template identifiers
  c.template_hash,
  c.template_name,
  SUBSTR(c.template_hash, 8, 12) as template_short_hash,
  
  -- Experiment dimensions
  c.model,
  c.criteria,
  c.record_hash,
  c.agent_role,
  
  -- Completeness metrics
  c.actual_predictions,
  c.session_count,
  c.first_run,
  c.last_run,
  c.completeness_status,
  c.sample_sessions,
  
  -- Template metadata for comparison context
  tm.criteria_count as template_criteria_count,
  tm.model_count as template_model_count,
  tm.total_predictions as template_total_predictions,
  tm.template_first_used,
  tm.template_last_used,
  
  -- Dynamic template labeling (based on usage frequency within role+criteria)
  CASE 
    WHEN ROW_NUMBER() OVER (
      PARTITION BY c.agent_role, c.criteria 
      ORDER BY tm.total_predictions DESC
    ) = 1 THEN 'Template A (Primary)'
    WHEN ROW_NUMBER() OVER (
      PARTITION BY c.agent_role, c.criteria 
      ORDER BY tm.total_predictions DESC  
    ) = 2 THEN 'Template B (Secondary)'
    ELSE CONCAT('Template ', ROW_NUMBER() OVER (
      PARTITION BY c.agent_role, c.criteria 
      ORDER BY tm.total_predictions DESC
    ))
  END as template_label,
  
  -- Experiment grouping (for filtering)
  CONCAT(c.agent_role, ' - ', c.criteria) as experiment_group

FROM completeness_analysis c
LEFT JOIN template_metadata tm USING (template_hash, template_name, agent_role)
ORDER BY c.agent_role, c.criteria, tm.total_predictions DESC, c.template_hash