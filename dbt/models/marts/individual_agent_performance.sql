{{ config(
    materialized='table',
    description='Individual agent performance analysis for JUDGE and SYNTHESISER agents with session-level aggregation'
) }}

-- Session-level performance analysis for individual agents
-- Properly handles multiple SCORERs per prediction and multiple agents per session
-- Enables detailed individual agent analysis and comparison

WITH experiment_labels AS (
  SELECT 
    template_hash,
    experiment_name,
    experiment_date,
    problem,
    change_link,
    description,
    success_metric
  FROM {{ ref('experiment_metadata') }}
),

-- Base data from judge_scores with proper deduplication
predictions_with_scores AS (
  SELECT DISTINCT
    js.session_id,
    js.call_id,
    js.record_id,
    js.timestamp,
    js.judge_model as agent_model,
    js.judge_criteria as criteria,
    js.judge_role as agent_role,
    js.judge_hash as template_hash,
    js.judge_template as template_name,
    js.predicted_violating,
    js.expected_violating,
    js.correct,
    js.confidence,
    js.full_reasons,
    js.tracing_link,
    
    -- SCORER evaluations (multiple per prediction)
    js.correctness,
    js.score_quality,
    js.scorer,
    js.scorer_model,
    js.scorer_call_id
    
  FROM {{ ref('judge_scores') }} js
  WHERE js.judge_role IN ('JUDGE', 'SYNTHESISER')
    AND js.timestamp >= '{{ var("cutoff_date") }}'
    AND js.score_quality = 'valid'  -- Only include valid scorer evaluations
),

-- Session-level agent performance aggregation
session_agent_performance AS (
  SELECT
    session_id,
    record_id,
    criteria,
    agent_model,
    agent_role,
    template_hash,
    template_name,
    
    -- Core performance metrics
    COUNT(DISTINCT call_id) as prediction_count,
    COUNT(DISTINCT scorer_call_id) as total_scorer_evaluations,
    
    -- Accuracy metrics (aggregated across multiple SCORERs)
    AVG(correctness) as avg_accuracy,
    STDDEV(correctness) as accuracy_stddev,
    MIN(correctness) as min_accuracy, 
    MAX(correctness) as max_accuracy,
    APPROX_QUANTILES(correctness, 2)[OFFSET(1)] as median_accuracy,
    
    -- Binary prediction accuracy
    AVG(CAST(CAST(correct AS INT64) AS FLOAT64)) as binary_accuracy,
    STDDEV(CAST(CAST(correct AS INT64) AS FLOAT64)) as binary_accuracy_stddev,
    
    -- Confidence metrics
    AVG(SAFE_CAST(confidence AS FLOAT64)) as avg_confidence,
    STDDEV(SAFE_CAST(confidence AS FLOAT64)) as confidence_stddev,
    
    -- Prediction patterns
    AVG(CAST(CAST(predicted_violating AS INT64) AS FLOAT64)) as violation_rate,
    
    -- Data quality indicators
    COUNT(DISTINCT scorer_model) as unique_scorer_models,
    MIN(timestamp) as first_prediction,
    MAX(timestamp) as last_prediction,
    
    -- Sample data for debugging
    ARRAY_AGG(DISTINCT call_id LIMIT 5) as sample_call_ids
    
  FROM predictions_with_scores
  GROUP BY 1, 2, 3, 4, 5, 6, 7
),

-- Add experiment metadata and derived metrics
agent_performance_enriched AS (
  SELECT
    sap.*,
    
    -- Experiment metadata
    el.experiment_name,
    el.experiment_date,
    el.problem as experiment_problem,
    el.change_link,
    el.description as experiment_description,
    el.success_metric,
    
    -- Performance quality indicators
    CASE 
      WHEN sap.prediction_count >= 10 THEN 'SUFFICIENT'
      WHEN sap.prediction_count >= 5 THEN 'LIMITED'
      ELSE 'INSUFFICIENT'
    END as data_sufficiency,
    
    CASE 
      WHEN sap.accuracy_stddev <= 0.1 THEN 'HIGH'
      WHEN sap.accuracy_stddev <= 0.2 THEN 'MEDIUM'
      ELSE 'LOW'
    END as accuracy_consistency,
    
    CASE 
      WHEN sap.confidence_stddev <= 0.1 THEN 'CONSISTENT'
      WHEN sap.confidence_stddev <= 0.2 THEN 'MODERATE'
      ELSE 'VARIABLE'
    END as confidence_consistency,
    
    -- Time-based metrics
    TIMESTAMP_DIFF(sap.last_prediction, sap.first_prediction, MINUTE) as prediction_span_minutes,
    DATE_DIFF(CURRENT_DATE(), DATE(sap.first_prediction), DAY) as days_since_first_prediction,
    
    -- Agent type categorization
    CONCAT(sap.agent_role, ' - ', sap.criteria) as agent_experiment_group,
    CONCAT(sap.agent_model, ' (', sap.agent_role, ')') as model_agent_label,
    
    -- Experiment labeling (with fallback)
    COALESCE(el.experiment_name, 
      CASE 
        WHEN ROW_NUMBER() OVER (
          PARTITION BY sap.agent_role, sap.criteria 
          ORDER BY sap.prediction_count DESC
        ) = 1 THEN 'Primary Template'
        ELSE CONCAT('Template ', ROW_NUMBER() OVER (
          PARTITION BY sap.agent_role, sap.criteria 
          ORDER BY sap.prediction_count DESC
        ))
      END
    ) as experiment_label
    
  FROM session_agent_performance sap
  LEFT JOIN experiment_labels el ON sap.template_hash = el.template_hash
),

-- Add ranking and comparison metrics
final_performance AS (
  SELECT 
    *,
    
    -- Performance rankings within experiment groups
    ROW_NUMBER() OVER (
      PARTITION BY agent_experiment_group, agent_model 
      ORDER BY avg_accuracy DESC
    ) as accuracy_rank_in_group,
    
    ROW_NUMBER() OVER (
      PARTITION BY agent_experiment_group, agent_model 
      ORDER BY binary_accuracy DESC
    ) as binary_accuracy_rank_in_group,
    
    -- Overall performance rankings
    PERCENT_RANK() OVER (
      PARTITION BY agent_role 
      ORDER BY avg_accuracy
    ) as accuracy_percentile_within_role,
    
    -- Performance flags
    CASE 
      WHEN avg_accuracy >= 0.8 THEN 'HIGH_PERFORMING'
      WHEN avg_accuracy >= 0.6 THEN 'MODERATE_PERFORMING'
      ELSE 'LOW_PERFORMING'
    END as performance_category
    
  FROM agent_performance_enriched
)

SELECT * FROM final_performance
ORDER BY 
  agent_role, 
  criteria, 
  agent_model, 
  avg_accuracy DESC