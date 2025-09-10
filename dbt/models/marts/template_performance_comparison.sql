{{ config(
    materialized='table',
    description='Template A/B testing performance comparison with accuracy metrics'
) }}

-- Get predictions with their accuracy scores from SCORER evaluations
WITH predictions_with_scores AS (
  SELECT DISTINCT
    p.call_id,
    p.template_hash,
    p.model,
    JSON_EXTRACT_SCALAR(p.metadata, '$.criteria') as criteria,
    JSON_EXTRACT_SCALAR(p.metadata, '$.record_hash') as record_hash,
    p.agent_name,
    p.timestamp,
    
    -- Extract accuracy from SCORER evaluations
    AVG(CAST(JSON_EXTRACT_SCALAR(s.output, '$.accuracy') AS FLOAT64)) as accuracy_score,
    AVG(CAST(JSON_EXTRACT_SCALAR(s.output, '$.score') AS FLOAT64)) as overall_score,
    COUNT(DISTINCT s.call_id) as num_scorers
    
  FROM {{ ref('stg_flows') }} p
  LEFT JOIN {{ ref('stg_flows') }} s ON p.call_id = s.parent_call_id AND s.agent_name = 'SCORER'
  WHERE p.agent_name IN ('JUDGE', 'SYNTH')
    AND p.template_hash IS NOT NULL
    AND JSON_EXTRACT_SCALAR(p.metadata, '$.record_hash') IS NOT NULL
    AND JSON_EXTRACT_SCALAR(p.metadata, '$.criteria') IS NOT NULL
  GROUP BY 1, 2, 3, 4, 5, 6, 7
),

-- Add template labels
predictions_labeled AS (
  SELECT *,
    ROW_NUMBER() OVER (ORDER BY template_hash) as template_rank,
    CASE 
      WHEN template_hash = (SELECT MIN(template_hash) FROM predictions_with_scores) THEN 'Template A'
      ELSE 'Template B'
    END as template_label
  FROM predictions_with_scores
),

-- Calculate performance by template and dimensions
template_performance AS (
  SELECT
    template_hash,
    template_label,
    model,
    criteria,
    agent_name,
    
    -- Core metrics
    COUNT(DISTINCT call_id) as total_predictions,
    COUNT(DISTINCT record_hash) as records_tested,
    AVG(accuracy_score) as avg_accuracy,
    STDDEV(accuracy_score) as accuracy_stddev,
    MIN(accuracy_score) as min_accuracy,
    MAX(accuracy_score) as max_accuracy,
    PERCENTILE_CONT(accuracy_score, 0.5) as median_accuracy,
    
    -- Overall score metrics
    AVG(overall_score) as avg_overall_score,
    STDDEV(overall_score) as overall_score_stddev,
    
    -- Data quality indicators
    AVG(num_scorers) as avg_scorers_per_prediction,
    MIN(timestamp) as first_prediction,
    MAX(timestamp) as last_prediction
    
  FROM predictions_labeled
  WHERE accuracy_score IS NOT NULL
  GROUP BY 1, 2, 3, 4, 5
),

-- Calculate lift between templates
template_comparison AS (
  SELECT
    model,
    criteria,
    agent_name,
    
    -- Template A metrics
    MAX(CASE WHEN template_label = 'Template A' THEN avg_accuracy END) as template_a_accuracy,
    MAX(CASE WHEN template_label = 'Template A' THEN total_predictions END) as template_a_predictions,
    MAX(CASE WHEN template_label = 'Template A' THEN accuracy_stddev END) as template_a_stddev,
    
    -- Template B metrics  
    MAX(CASE WHEN template_label = 'Template B' THEN avg_accuracy END) as template_b_accuracy,
    MAX(CASE WHEN template_label = 'Template B' THEN total_predictions END) as template_b_predictions,
    MAX(CASE WHEN template_label = 'Template B' THEN accuracy_stddev END) as template_b_stddev,
    
    -- Calculate lift and significance
    (MAX(CASE WHEN template_label = 'Template B' THEN avg_accuracy END) - 
     MAX(CASE WHEN template_label = 'Template A' THEN avg_accuracy END)) as absolute_lift,
    
    CASE 
      WHEN MAX(CASE WHEN template_label = 'Template A' THEN avg_accuracy END) > 0 THEN
        ((MAX(CASE WHEN template_label = 'Template B' THEN avg_accuracy END) - 
          MAX(CASE WHEN template_label = 'Template A' THEN avg_accuracy END)) / 
         MAX(CASE WHEN template_label = 'Template A' THEN avg_accuracy END)) * 100
      ELSE NULL
    END as relative_lift_percent,
    
    -- Data completeness check
    CASE 
      WHEN MAX(CASE WHEN template_label = 'Template A' THEN total_predictions END) >= 10 
       AND MAX(CASE WHEN template_label = 'Template B' THEN total_predictions END) >= 10 
      THEN 'SUFFICIENT'
      ELSE 'INSUFFICIENT'
    END as data_sufficiency
    
  FROM template_performance
  GROUP BY 1, 2, 3
)

-- Final output combining both views
SELECT
  -- Dimensions
  tp.template_hash,
  tp.template_label,
  tp.model,
  tp.criteria,
  tp.agent_name,
  
  -- Performance metrics
  tp.total_predictions,
  tp.records_tested,
  tp.avg_accuracy,
  tp.accuracy_stddev,
  tp.median_accuracy,
  tp.min_accuracy,
  tp.max_accuracy,
  tp.avg_overall_score,
  tp.overall_score_stddev,
  
  -- Comparison metrics (from template_comparison)
  tc.absolute_lift,
  tc.relative_lift_percent,
  tc.data_sufficiency,
  
  -- Confidence indicators
  CASE 
    WHEN tp.accuracy_stddev < 0.1 THEN 'HIGH'
    WHEN tp.accuracy_stddev < 0.2 THEN 'MEDIUM' 
    ELSE 'LOW'
  END as confidence_level,
  
  -- Data quality
  tp.avg_scorers_per_prediction,
  tp.first_prediction,
  tp.last_prediction,
  
  -- Rankings within criteria/model
  ROW_NUMBER() OVER (
    PARTITION BY tp.model, tp.criteria, tp.agent_name 
    ORDER BY tp.avg_accuracy DESC
  ) as accuracy_rank

FROM template_performance tp
LEFT JOIN template_comparison tc ON (
  tp.model = tc.model 
  AND tp.criteria = tc.criteria 
  AND tp.agent_name = tc.agent_name
)
ORDER BY tp.model, tp.criteria, tp.agent_name, tp.template_label