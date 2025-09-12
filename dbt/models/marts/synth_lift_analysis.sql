{{ config(
    materialized='table',
    description='Session-based synth lift analysis comparing JUDGE vs SYNTHESISER performance with statistical measures'
) }}

-- Synth lift analysis: quantifies the value added by SYNTHESISER agents
-- compared to JUDGE agents within the same sessions
-- Uses proper session-based joins as documented in eval.md

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

-- Session-level performance for JUDGE agents
judge_session_performance AS (
  SELECT
    iap.session_id,
    iap.record_id,
    iap.criteria,
    iap.agent_model,
    iap.template_hash,
    iap.experiment_name,
    iap.experiment_date,
    
    -- JUDGE performance metrics (averaged across multiple JUDGEs if present)
    AVG(iap.avg_accuracy) as judge_avg_accuracy,
    AVG(iap.binary_accuracy) as judge_binary_accuracy,
    AVG(iap.avg_confidence) as judge_avg_confidence,
    AVG(iap.violation_rate) as judge_violation_rate,
    
    -- JUDGE consistency metrics
    STDDEV(iap.avg_accuracy) as judge_accuracy_variance,
    STDDEV(iap.avg_confidence) as judge_confidence_variance,
    
    -- JUDGE data quality
    SUM(iap.prediction_count) as judge_total_predictions,
    COUNT(*) as judge_agent_count,
    MIN(iap.first_prediction) as judge_first_prediction,
    MAX(iap.last_prediction) as judge_last_prediction,
    
    -- JUDGE performance consistency
    CASE 
      WHEN STDDEV(iap.avg_accuracy) <= 0.1 THEN 'CONSISTENT'
      WHEN STDDEV(iap.avg_accuracy) <= 0.2 THEN 'MODERATE'
      ELSE 'VARIABLE'
    END as judge_consistency,
    
    ARRAY_AGG(DISTINCT iap.template_hash IGNORE NULLS) as judge_template_hashes
    
  FROM {{ ref('individual_agent_performance') }} iap
  WHERE iap.agent_role = 'JUDGE'
    AND iap.data_sufficiency IN ('SUFFICIENT', 'LIMITED')  -- Only analyze quality data
  GROUP BY 1, 2, 3, 4, 5, 6, 7
),

-- Session-level performance for SYNTHESISER agents  
synth_session_performance AS (
  SELECT
    iap.session_id,
    iap.record_id,
    iap.criteria,
    iap.agent_model,
    iap.template_hash,
    iap.experiment_name,
    iap.experiment_date,
    
    -- SYNTH performance metrics (averaged across multiple SYNTHs if present)
    AVG(iap.avg_accuracy) as synth_avg_accuracy,
    AVG(iap.binary_accuracy) as synth_binary_accuracy,
    AVG(iap.avg_confidence) as synth_avg_confidence,
    AVG(iap.violation_rate) as synth_violation_rate,
    
    -- SYNTH consistency metrics
    STDDEV(iap.avg_accuracy) as synth_accuracy_variance,
    STDDEV(iap.avg_confidence) as synth_confidence_variance,
    
    -- SYNTH data quality
    SUM(iap.prediction_count) as synth_total_predictions,
    COUNT(*) as synth_agent_count,
    MIN(iap.first_prediction) as synth_first_prediction,
    MAX(iap.last_prediction) as synth_last_prediction,
    
    -- SYNTH performance consistency
    CASE 
      WHEN STDDEV(iap.avg_accuracy) <= 0.1 THEN 'CONSISTENT'
      WHEN STDDEV(iap.avg_accuracy) <= 0.2 THEN 'MODERATE'
      ELSE 'VARIABLE'
    END as synth_consistency,
    
    ARRAY_AGG(DISTINCT iap.template_hash IGNORE NULLS) as synth_template_hashes
    
  FROM {{ ref('individual_agent_performance') }} iap
  WHERE iap.agent_role = 'SYNTHESISER'
    AND iap.data_sufficiency IN ('SUFFICIENT', 'LIMITED')  -- Only analyze quality data
  GROUP BY 1, 2, 3, 4, 5, 6, 7
),

-- Session-based lift calculations with statistical measures
synth_lift_calculations AS (
  SELECT
    -- Session identifiers
    j.session_id,
    j.record_id,
    j.criteria,
    j.agent_model,
    
    -- Experiment metadata (prioritize non-null values)
    COALESCE(j.experiment_name, s.experiment_name) as experiment_name,
    COALESCE(j.experiment_date, s.experiment_date) as experiment_date,
    
    -- Core performance comparison
    j.judge_avg_accuracy,
    s.synth_avg_accuracy,
    j.judge_binary_accuracy,
    s.synth_binary_accuracy,
    
    -- Lift calculations (accuracy-based)
    (s.synth_avg_accuracy - j.judge_avg_accuracy) as absolute_accuracy_lift,
    CASE 
      WHEN j.judge_avg_accuracy > 0 THEN
        ((s.synth_avg_accuracy - j.judge_avg_accuracy) / j.judge_avg_accuracy) * 100
      ELSE NULL
    END as relative_accuracy_lift_pct,
    
    -- Binary accuracy lift
    (s.synth_binary_accuracy - j.judge_binary_accuracy) as absolute_binary_lift,
    CASE 
      WHEN j.judge_binary_accuracy > 0 THEN
        ((s.synth_binary_accuracy - j.judge_binary_accuracy) / j.judge_binary_accuracy) * 100
      ELSE NULL
    END as relative_binary_lift_pct,
    
    -- Confidence comparison
    j.judge_avg_confidence,
    s.synth_avg_confidence,
    (s.synth_avg_confidence - j.judge_avg_confidence) as confidence_lift,
    
    -- Prediction pattern comparison
    j.judge_violation_rate,
    s.synth_violation_rate,
    (s.synth_violation_rate - j.judge_violation_rate) as violation_rate_diff,
    
    -- Consistency comparison
    j.judge_consistency,
    s.synth_consistency,
    j.judge_accuracy_variance,
    s.synth_accuracy_variance,
    
    -- Data quality metrics
    j.judge_total_predictions,
    s.synth_total_predictions,
    j.judge_agent_count,
    s.synth_agent_count,
    
    -- Timing analysis
    j.judge_first_prediction,
    j.judge_last_prediction,
    s.synth_first_prediction,
    s.synth_last_prediction,
    
    TIMESTAMP_DIFF(s.synth_first_prediction, j.judge_last_prediction, MINUTE) as synth_delay_minutes,
    
    -- Template diversity
    j.judge_template_hashes,
    s.synth_template_hashes
    
  FROM judge_session_performance j
  INNER JOIN synth_session_performance s 
    ON j.session_id = s.session_id 
    AND j.record_id = s.record_id 
    AND j.criteria = s.criteria
    AND j.agent_model = s.agent_model  -- Compare same model across agent types
),

-- Add statistical significance and categorization
lift_analysis_final AS (
  SELECT
    *,
    
    -- Lift categorization (accuracy-based)
    CASE 
      WHEN absolute_accuracy_lift >= 0.15 THEN 'HIGH_POSITIVE_LIFT'
      WHEN absolute_accuracy_lift >= 0.05 THEN 'MODERATE_POSITIVE_LIFT'
      WHEN absolute_accuracy_lift >= -0.05 THEN 'NEUTRAL_LIFT'
      WHEN absolute_accuracy_lift >= -0.15 THEN 'MODERATE_NEGATIVE_LIFT'
      ELSE 'HIGH_NEGATIVE_LIFT'
    END as lift_category,
    
    -- Effect size classification
    CASE 
      WHEN ABS(absolute_accuracy_lift) >= 0.2 THEN 'LARGE_EFFECT'
      WHEN ABS(absolute_accuracy_lift) >= 0.1 THEN 'MEDIUM_EFFECT'
      WHEN ABS(absolute_accuracy_lift) >= 0.02 THEN 'SMALL_EFFECT'
      ELSE 'NEGLIGIBLE_EFFECT'
    END as effect_size,
    
    -- Statistical confidence (based on sample sizes)
    CASE 
      WHEN judge_total_predictions >= 30 AND synth_total_predictions >= 30 THEN 'HIGH_CONFIDENCE'
      WHEN judge_total_predictions >= 10 AND synth_total_predictions >= 10 THEN 'MEDIUM_CONFIDENCE'
      WHEN judge_total_predictions >= 5 AND synth_total_predictions >= 5 THEN 'LOW_CONFIDENCE'
      ELSE 'INSUFFICIENT_DATA'
    END as statistical_confidence,
    
    -- Workflow efficiency assessment
    CASE 
      WHEN absolute_accuracy_lift > 0.1 AND synth_delay_minutes <= 30 THEN 'EFFICIENT_IMPROVEMENT'
      WHEN absolute_accuracy_lift > 0.05 THEN 'WORTHWHILE_IMPROVEMENT'
      WHEN absolute_accuracy_lift >= -0.02 THEN 'NEUTRAL_EFFICIENCY'
      ELSE 'INEFFICIENT_DEGRADATION'
    END as workflow_efficiency,
    
    -- Consistency impact
    CASE 
      WHEN judge_consistency = 'VARIABLE' AND synth_consistency IN ('CONSISTENT', 'MODERATE') THEN 'CONSISTENCY_IMPROVED'
      WHEN judge_consistency IN ('CONSISTENT', 'MODERATE') AND synth_consistency = 'VARIABLE' THEN 'CONSISTENCY_DEGRADED'
      ELSE 'CONSISTENCY_MAINTAINED'
    END as consistency_impact,
    
    -- Overall synth value assessment
    CASE 
      WHEN absolute_accuracy_lift >= 0.1 THEN 'HIGH_VALUE'
      WHEN absolute_accuracy_lift >= 0.05 OR 
           (absolute_accuracy_lift >= 0.02 AND synth_accuracy_variance < judge_accuracy_variance) THEN 'MODERATE_VALUE'
      WHEN absolute_accuracy_lift >= -0.02 THEN 'LOW_VALUE'
      ELSE 'NEGATIVE_VALUE'
    END as synth_value_assessment
    
  FROM synth_lift_calculations
)

SELECT * FROM lift_analysis_final
ORDER BY 
  experiment_name NULLS LAST,
  criteria,
  agent_model,
  absolute_accuracy_lift DESC