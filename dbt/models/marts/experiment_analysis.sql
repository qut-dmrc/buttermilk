{{ config(
    materialized='view',
    description='Experiment analysis with meaningful labels and metadata for dashboard consumption'
) }}

-- Simplified view for dashboard queries with experiment context
WITH experiments AS (
  SELECT * FROM {{ ref('experiment_metadata') }}
),

performance AS (
  SELECT * FROM {{ ref('template_performance_comparison') }}
)

SELECT 
  -- Template and experiment identification
  p.template_hash,
  p.template_name,
  p.template_short_hash,
  p.template_label,
  p.experiment_group,
  
  -- Experiment metadata with calculated fields
  p.experiment_name,
  p.experiment_date,
  p.experiment_problem,
  p.change_link,
  p.experiment_description,
  p.success_metric,
  
  -- Calculate days since experiment
  CASE 
    WHEN p.experiment_date IS NOT NULL THEN 
      DATE_DIFF(CURRENT_DATE(), p.experiment_date, DAY)
    ELSE NULL
  END as days_since_experiment,
  
  -- Experiment categorization
  CASE 
    WHEN p.experiment_name IS NOT NULL THEN 'LABELED_EXPERIMENT'
    ELSE 'UNLABELED_DATA'
  END as experiment_status,
  
  -- Core experiment dimensions
  p.model,
  p.criteria,
  p.agent_role,
  
  -- Performance metrics
  p.total_predictions,
  p.session_count,
  p.records_tested,
  p.avg_accuracy,
  p.accuracy_stddev,
  p.median_accuracy,
  p.min_accuracy,
  p.max_accuracy,
  p.data_sufficiency,
  p.confidence_level,
  
  -- Comparison metrics (for A/B testing)
  p.absolute_lift,
  p.relative_lift_percent,
  p.statistical_confidence,
  p.effect_size,
  
  -- Data quality and timing
  p.avg_scorers_per_prediction,
  p.first_prediction,
  p.last_prediction,
  p.accuracy_rank_in_group,
  
  -- Dashboard-friendly fields
  CONCAT(
    COALESCE(p.experiment_name, 'Unlabeled'), 
    ' (', 
    FORMAT('%d', p.total_predictions), 
    ' predictions)'
  ) as display_label,
  
  -- Success indicator
  CASE 
    WHEN p.success_metric IS NOT NULL AND p.success_metric != '' THEN 'SUCCESS_DOCUMENTED'
    WHEN p.experiment_name IS NOT NULL THEN 'SUCCESS_PENDING'
    ELSE 'NOT_APPLICABLE'
  END as success_status

FROM performance p
ORDER BY 
  p.experiment_date DESC NULLS LAST,
  p.experiment_group,
  p.total_predictions DESC