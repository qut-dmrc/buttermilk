{{ config(
    materialized='table',
    description='Data completeness validation for template A/B testing experiment'
) }}

-- Identify all expected experimental combinations and check for gaps
WITH experiment_dimensions AS (
  SELECT DISTINCT
    template_hash,
    model,
    JSON_EXTRACT_SCALAR(metadata, '$.criteria') as criteria,
    JSON_EXTRACT_SCALAR(metadata, '$.record_hash') as record_hash,
    agent_name
  FROM {{ ref('stg_flows') }}
  WHERE agent_name IN ('JUDGE', 'SYNTH')
    AND template_hash IS NOT NULL
    AND JSON_EXTRACT_SCALAR(metadata, '$.record_hash') IS NOT NULL
    AND JSON_EXTRACT_SCALAR(metadata, '$.criteria') IS NOT NULL
),

-- Get actual predictions
actual_predictions AS (
  SELECT DISTINCT
    template_hash,
    model,
    JSON_EXTRACT_SCALAR(metadata, '$.criteria') as criteria,
    JSON_EXTRACT_SCALAR(metadata, '$.record_hash') as record_hash,
    agent_name,
    COUNT(DISTINCT call_id) as prediction_count,
    MIN(timestamp) as first_run,
    MAX(timestamp) as last_run
  FROM {{ ref('stg_flows') }}
  WHERE agent_name IN ('JUDGE', 'SYNTH')
    AND template_hash IS NOT NULL
    AND JSON_EXTRACT_SCALAR(metadata, '$.record_hash') IS NOT NULL
    AND JSON_EXTRACT_SCALAR(metadata, '$.criteria') IS NOT NULL
  GROUP BY 1, 2, 3, 4, 5
),

-- Cross-join to get all expected combinations
expected_combinations AS (
  SELECT DISTINCT
    t.template_hash,
    m.model,
    c.criteria,
    r.record_hash,
    a.agent_name
  FROM (SELECT DISTINCT template_hash FROM experiment_dimensions) t
  CROSS JOIN (SELECT DISTINCT model FROM experiment_dimensions) m
  CROSS JOIN (SELECT DISTINCT criteria FROM experiment_dimensions) c
  CROSS JOIN (SELECT DISTINCT record_hash FROM experiment_dimensions) r
  CROSS JOIN (SELECT DISTINCT agent_name FROM experiment_dimensions) a
),

-- Identify gaps
completeness_analysis AS (
  SELECT
    e.template_hash,
    e.model,
    e.criteria,
    e.record_hash,
    e.agent_name,
    COALESCE(a.prediction_count, 0) as actual_predictions,
    a.first_run,
    a.last_run,
    CASE 
      WHEN a.prediction_count IS NULL THEN 'MISSING'
      WHEN a.prediction_count < 10 THEN 'INSUFFICIENT' -- Need 10+ for stochastic testing
      ELSE 'COMPLETE'
    END as completeness_status
  FROM expected_combinations e
  LEFT JOIN actual_predictions a USING (template_hash, model, criteria, record_hash, agent_name)
)

SELECT
  template_hash,
  model,
  criteria,
  record_hash,
  agent_name,
  actual_predictions,
  first_run,
  last_run,
  completeness_status,
  
  -- Add template labels for readability
  CASE 
    WHEN template_hash = LAG(template_hash) OVER (ORDER BY template_hash) THEN 'Template B'
    ELSE 'Template A'
  END as template_label

FROM completeness_analysis
ORDER BY template_hash, model, criteria, record_hash, agent_name