{{ config(
    materialized='table',
    description='Template hash-based performance comparison - filter by template_hash to define experiments'
) }}

-- This model compares performance between different template hashes
-- Use dashboard filters to select which template hashes to compare for your experiment

WITH predictions_with_scores AS (
  SELECT DISTINCT
    p.template_hash,
    p.template_name,
    p.call_id,
    p.agent_model as model,
    p.judge_criteria as criteria,
    p.record_id as record_hash,
    p.agent_role,
    p.session_id,
    p.timestamp,
    
    -- Extract accuracy from SCORER evaluations  
    AVG(s.correctness) as accuracy_score,
    COUNT(DISTINCT s.call_id) as num_scorers
    
  FROM {{ ref('stg_flows') }} p
  LEFT JOIN {{ ref('stg_flows') }} s ON p.call_id = s.parent_call_id AND s.agent_role = 'SCORERS'
  WHERE p.agent_role IN ('JUDGE', 'SYNTHESISER')
    AND p.template_hash IS NOT NULL
    AND p.record_id IS NOT NULL
    AND p.judge_criteria IS NOT NULL
  GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
),

-- Calculate performance metrics for each template hash
template_performance AS (
  SELECT
    template_hash,
    template_name,
    SUBSTR(template_hash, 8, 12) as template_short_hash,
    model,
    criteria,
    agent_role,
    
    -- Core performance metrics
    COUNT(DISTINCT call_id) as total_predictions,
    COUNT(DISTINCT session_id) as session_count,
    COUNT(DISTINCT record_hash) as records_tested,
    AVG(accuracy_score) as avg_accuracy,
    STDDEV(accuracy_score) as accuracy_stddev,
    MIN(accuracy_score) as min_accuracy,
    MAX(accuracy_score) as max_accuracy,
    APPROX_QUANTILES(accuracy_score, 2)[OFFSET(1)] as median_accuracy,
    
    -- Data quality indicators
    AVG(num_scorers) as avg_scorers_per_prediction,
    MIN(timestamp) as first_prediction,
    MAX(timestamp) as last_prediction,
    
    -- Template usage metadata
    ARRAY_AGG(DISTINCT session_id LIMIT 10) as sample_sessions
    
  FROM predictions_with_scores
  WHERE accuracy_score IS NOT NULL
  GROUP BY 1, 2, 3, 4, 5, 6
),

-- Join with experiment metadata and add template labeling
experiment_labels AS (
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

-- Add dynamic template labeling within each experiment group
labeled_performance AS (
  SELECT tp.*,
    -- Use experiment name if available, otherwise fall back to dynamic labeling
    COALESCE(el.experiment_name, 
      CASE 
        WHEN ROW_NUMBER() OVER (
          PARTITION BY tp.agent_role, tp.criteria 
          ORDER BY tp.total_predictions DESC
        ) = 1 THEN 'Template A (Primary)'
        WHEN ROW_NUMBER() OVER (
          PARTITION BY tp.agent_role, tp.criteria 
          ORDER BY tp.total_predictions DESC
        ) = 2 THEN 'Template B (Secondary)'
        ELSE CONCAT('Template ', ROW_NUMBER() OVER (
          PARTITION BY tp.agent_role, tp.criteria 
          ORDER BY tp.total_predictions DESC
        ))
      END
    ) as template_label,
    
    -- Add experiment metadata
    el.experiment_name,
    el.experiment_date,
    el.problem as experiment_problem,
    el.change_link,
    el.description as experiment_description,
    el.success_metric,
    
    -- Create experiment group identifier
    CONCAT(agent_role, ' - ', criteria) as experiment_group,
    
    -- Data sufficiency flag
    CASE 
      WHEN total_predictions >= 10 THEN 'SUFFICIENT'
      ELSE 'INSUFFICIENT'
    END as data_sufficiency,
    
    -- Confidence level based on variance
    CASE 
      WHEN tp.accuracy_stddev < 0.1 THEN 'HIGH'
      WHEN tp.accuracy_stddev < 0.2 THEN 'MEDIUM' 
      ELSE 'LOW'
    END as confidence_level
    
  FROM template_performance tp
  LEFT JOIN experiment_labels el ON tp.template_hash = el.template_hash
),

-- Calculate pairwise comparisons (lift) between templates
template_comparisons AS (
  SELECT 
    a.experiment_group,
    a.model,
    a.criteria,
    a.agent_role,
    
    -- Template A (Primary) metrics
    a.template_hash as template_a_hash,
    a.template_short_hash as template_a_short,
    a.template_label as template_a_label,
    a.avg_accuracy as template_a_accuracy,
    a.total_predictions as template_a_predictions,
    a.data_sufficiency as template_a_sufficiency,
    
    -- Template B (Secondary) metrics  
    b.template_hash as template_b_hash,
    b.template_short_hash as template_b_short,
    b.template_label as template_b_label,
    b.avg_accuracy as template_b_accuracy,
    b.total_predictions as template_b_predictions,
    b.data_sufficiency as template_b_sufficiency,
    
    -- Lift calculations
    (b.avg_accuracy - a.avg_accuracy) as absolute_lift,
    CASE 
      WHEN a.avg_accuracy > 0 THEN
        ((b.avg_accuracy - a.avg_accuracy) / a.avg_accuracy) * 100
      ELSE NULL
    END as relative_lift_percent,
    
    -- Statistical confidence
    CASE 
      WHEN a.total_predictions >= 30 AND b.total_predictions >= 30 THEN 'HIGH_CONFIDENCE'
      WHEN a.total_predictions >= 10 AND b.total_predictions >= 10 THEN 'MEDIUM_CONFIDENCE'
      ELSE 'LOW_CONFIDENCE'
    END as statistical_confidence,
    
    -- Effect size
    CASE 
      WHEN ABS(b.avg_accuracy - a.avg_accuracy) >= 0.1 THEN 'LARGE_EFFECT'
      WHEN ABS(b.avg_accuracy - a.avg_accuracy) >= 0.05 THEN 'MEDIUM_EFFECT'
      WHEN ABS(b.avg_accuracy - a.avg_accuracy) >= 0.01 THEN 'SMALL_EFFECT'
      ELSE 'NEGLIGIBLE_EFFECT'
    END as effect_size
    
  FROM labeled_performance a
  LEFT JOIN labeled_performance b ON (
    a.experiment_group = b.experiment_group
    AND a.model = b.model
    AND a.template_label = 'Template A (Primary)'
    AND b.template_label = 'Template B (Secondary)'
  )
  WHERE a.template_label = 'Template A (Primary)'
)

-- Final output: Individual template performance + comparison metrics
SELECT
  -- Template identification
  lp.template_hash,
  lp.template_name,
  lp.template_short_hash,
  lp.template_label,
  lp.experiment_group,
  
  -- Experiment metadata
  lp.experiment_name,
  lp.experiment_date,
  lp.experiment_problem,
  lp.change_link,
  lp.experiment_description,
  lp.success_metric,
  
  -- Experiment dimensions
  lp.model,
  lp.criteria,
  lp.agent_role,
  
  -- Performance metrics
  lp.total_predictions,
  lp.session_count,
  lp.records_tested,
  lp.avg_accuracy,
  lp.accuracy_stddev,
  lp.median_accuracy,
  lp.min_accuracy,
  lp.max_accuracy,
  lp.data_sufficiency,
  lp.confidence_level,
  
  -- Comparison metrics (null for templates without pairs)
  tc.absolute_lift,
  tc.relative_lift_percent,
  tc.statistical_confidence,
  tc.effect_size,
  
  -- Comparison context
  tc.template_a_hash,
  tc.template_b_hash,
  tc.template_a_accuracy,
  tc.template_b_accuracy,
  
  -- Data quality
  lp.avg_scorers_per_prediction,
  lp.first_prediction,
  lp.last_prediction,
  lp.sample_sessions,
  
  -- Rankings
  ROW_NUMBER() OVER (
    PARTITION BY lp.experiment_group, lp.model 
    ORDER BY lp.avg_accuracy DESC
  ) as accuracy_rank_in_group

FROM labeled_performance lp
LEFT JOIN template_comparisons tc ON (
  lp.experiment_group = tc.experiment_group
  AND lp.model = tc.model
  AND lp.template_hash IN (tc.template_a_hash, tc.template_b_hash)
)
ORDER BY lp.experiment_group, lp.model, lp.total_predictions DESC, lp.template_hash