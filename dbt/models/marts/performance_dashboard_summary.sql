{{ config(
    materialized='table',
    description='Pre-aggregated performance summary metrics and KPIs optimized for dashboard loading and executive summary views'
) }}

-- Dashboard summary with key performance indicators and executive metrics
-- Pre-computed to minimize dashboard loading time

WITH overall_performance_summary AS (
  SELECT
    'Overall Performance' as metric_category,
    
    -- Volume metrics
    COUNT(DISTINCT session_id) as total_sessions,
    COUNT(DISTINCT CONCAT(session_id, record_id, criteria, agent_model)) as total_experiments,
    SUM(prediction_count) as total_predictions,
    SUM(total_scorer_evaluations) as total_scorer_evaluations,
    
    -- Performance metrics
    AVG(avg_accuracy) as overall_avg_accuracy,
    STDDEV(avg_accuracy) as overall_accuracy_stddev,
    MIN(avg_accuracy) as overall_min_accuracy,
    MAX(avg_accuracy) as overall_max_accuracy,
    
    -- Data quality
    COUNT(CASE WHEN data_sufficiency = 'SUFFICIENT' THEN 1 END) as sufficient_data_sessions,
    COUNT(CASE WHEN data_sufficiency = 'LIMITED' THEN 1 END) as limited_data_sessions,
    COUNT(CASE WHEN data_sufficiency = 'INSUFFICIENT' THEN 1 END) as insufficient_data_sessions,
    
    -- Performance distribution
    COUNT(CASE WHEN performance_category = 'HIGH_PERFORMING' THEN 1 END) as high_performing_sessions,
    COUNT(CASE WHEN performance_category = 'MODERATE_PERFORMING' THEN 1 END) as moderate_performing_sessions,
    COUNT(CASE WHEN performance_category = 'LOW_PERFORMING' THEN 1 END) as low_performing_sessions
    
  FROM {{ ref('individual_agent_performance') }}
  WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')
),

judge_vs_synth_summary AS (
  SELECT
    'Judge vs Synth' as metric_category,
    
    -- Agent type breakdown
    COUNT(CASE WHEN agent_role = 'JUDGE' THEN 1 END) as judge_sessions,
    COUNT(CASE WHEN agent_role = 'SYNTHESISER' THEN 1 END) as synth_sessions,
    
    -- Performance by agent type
    AVG(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END) as judge_avg_accuracy,
    AVG(CASE WHEN agent_role = 'SYNTHESISER' THEN avg_accuracy END) as synth_avg_accuracy,
    
    STDDEV(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END) as judge_accuracy_stddev,
    STDDEV(CASE WHEN agent_role = 'SYNTHESISER' THEN avg_accuracy END) as synth_accuracy_stddev,
    
    -- Simple lift calculation (overall averages)
    AVG(CASE WHEN agent_role = 'SYNTHESISER' THEN avg_accuracy END) - 
    AVG(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END) as simple_accuracy_lift,
    
    ((AVG(CASE WHEN agent_role = 'SYNTHESISER' THEN avg_accuracy END) - 
      AVG(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END)) /
     AVG(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END)) * 100 as simple_lift_percentage
    
  FROM {{ ref('individual_agent_performance') }}
  WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')
),

synth_lift_summary AS (
  SELECT
    'Synth Lift Analysis' as metric_category,
    
    -- Lift sessions analysis
    COUNT(*) as total_lift_sessions,
    
    -- Lift distribution
    COUNT(CASE WHEN lift_category LIKE '%POSITIVE%' THEN 1 END) as positive_lift_sessions,
    COUNT(CASE WHEN lift_category = 'NEUTRAL_LIFT' THEN 1 END) as neutral_lift_sessions,
    COUNT(CASE WHEN lift_category LIKE '%NEGATIVE%' THEN 1 END) as negative_lift_sessions,
    
    -- Lift magnitudes
    AVG(absolute_accuracy_lift) as avg_absolute_lift,
    AVG(relative_accuracy_lift_pct) as avg_relative_lift_pct,
    STDDEV(absolute_accuracy_lift) as lift_stddev,
    
    -- Best and worst lift
    MAX(absolute_accuracy_lift) as max_positive_lift,
    MIN(absolute_accuracy_lift) as max_negative_lift,
    
    -- Effect sizes
    COUNT(CASE WHEN effect_size = 'LARGE_EFFECT' THEN 1 END) as large_effect_sessions,
    COUNT(CASE WHEN effect_size = 'MEDIUM_EFFECT' THEN 1 END) as medium_effect_sessions,
    COUNT(CASE WHEN effect_size = 'SMALL_EFFECT' THEN 1 END) as small_effect_sessions,
    COUNT(CASE WHEN effect_size = 'NEGLIGIBLE_EFFECT' THEN 1 END) as negligible_effect_sessions,
    
    -- Value assessment
    COUNT(CASE WHEN synth_value_assessment = 'HIGH_VALUE' THEN 1 END) as high_value_sessions,
    COUNT(CASE WHEN synth_value_assessment = 'MODERATE_VALUE' THEN 1 END) as moderate_value_sessions,
    COUNT(CASE WHEN synth_value_assessment = 'LOW_VALUE' THEN 1 END) as low_value_sessions,
    COUNT(CASE WHEN synth_value_assessment = 'NEGATIVE_VALUE' THEN 1 END) as negative_value_sessions,
    
    -- Statistical confidence
    COUNT(CASE WHEN statistical_confidence = 'HIGH_CONFIDENCE' THEN 1 END) as high_confidence_sessions,
    COUNT(CASE WHEN statistical_confidence = 'MEDIUM_CONFIDENCE' THEN 1 END) as medium_confidence_sessions,
    COUNT(CASE WHEN statistical_confidence = 'LOW_CONFIDENCE' THEN 1 END) as low_confidence_sessions
    
  FROM {{ ref('synth_lift_analysis') }}
),

model_performance_summary AS (
  SELECT
    'Model Performance' as metric_category,
    agent_model,
    
    -- Performance by model
    COUNT(DISTINCT session_id) as model_sessions,
    AVG(avg_accuracy) as model_avg_accuracy,
    STDDEV(avg_accuracy) as model_accuracy_stddev,
    
    -- Model performance by agent type
    AVG(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END) as model_judge_accuracy,
    AVG(CASE WHEN agent_role = 'SYNTHESISER' THEN avg_accuracy END) as model_synth_accuracy,
    
    RANK() OVER (ORDER BY AVG(avg_accuracy) DESC) as model_accuracy_rank
    
  FROM {{ ref('individual_agent_performance') }}
  WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')
  GROUP BY agent_model
),

criteria_performance_summary AS (
  SELECT
    'Criteria Performance' as metric_category,
    criteria,
    
    -- Performance by criteria
    COUNT(DISTINCT session_id) as criteria_sessions,
    AVG(avg_accuracy) as criteria_avg_accuracy,
    STDDEV(avg_accuracy) as criteria_accuracy_stddev,
    
    -- Criteria performance by agent type
    AVG(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END) as criteria_judge_accuracy,
    AVG(CASE WHEN agent_role = 'SYNTHESISER' THEN avg_accuracy END) as criteria_synth_accuracy,
    
    RANK() OVER (ORDER BY AVG(avg_accuracy) DESC) as criteria_accuracy_rank
    
  FROM {{ ref('individual_agent_performance') }}
  WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')
  GROUP BY criteria
),

experiment_performance_summary AS (
  SELECT
    'Experiment Performance' as metric_category,
    COALESCE(experiment_name, 'Unlabeled') as experiment_name,
    
    -- Performance by experiment
    COUNT(DISTINCT session_id) as experiment_sessions,
    AVG(avg_accuracy) as experiment_avg_accuracy,
    STDDEV(avg_accuracy) as experiment_accuracy_stddev,
    
    -- Experiment performance by agent type
    AVG(CASE WHEN agent_role = 'JUDGE' THEN avg_accuracy END) as experiment_judge_accuracy,
    AVG(CASE WHEN agent_role = 'SYNTHESISER' THEN avg_accuracy END) as experiment_synth_accuracy,
    
    RANK() OVER (ORDER BY AVG(avg_accuracy) DESC) as experiment_accuracy_rank
    
  FROM {{ ref('individual_agent_performance') }}
  WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')
  GROUP BY experiment_name
),

-- Top performers identification
top_performers AS (
  SELECT
    'Top Performers' as metric_category,
    
    -- Best overall combinations
    (SELECT CONCAT(agent_model, ' (', agent_role, ') - ', criteria) 
     FROM {{ ref('individual_agent_performance') }}
     WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')
     ORDER BY avg_accuracy DESC 
     LIMIT 1) as best_overall_combination,
     
    (SELECT MAX(avg_accuracy)
     FROM {{ ref('individual_agent_performance') }}
     WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')
    ) as best_overall_accuracy,
    
    -- Best synth lift
    (SELECT CONCAT(agent_model, ' - ', criteria)
     FROM {{ ref('synth_lift_analysis') }}
     ORDER BY absolute_accuracy_lift DESC
     LIMIT 1) as best_synth_lift_combination,
     
    (SELECT MAX(absolute_accuracy_lift)
     FROM {{ ref('synth_lift_analysis') }}
    ) as best_synth_lift_value
),

-- Create final summary with calculated percentages
final_summary AS (
  SELECT
    'OVERALL_METRICS' as summary_type,
    metric_category,
    
    -- Core metrics (using JSON for structured storage)
    JSON_OBJECT(
      'total_sessions', total_sessions,
      'total_experiments', total_experiments,
      'total_predictions', total_predictions,
      'avg_accuracy', ROUND(overall_avg_accuracy, 3),
      'accuracy_stddev', ROUND(overall_accuracy_stddev, 3),
      'min_accuracy', ROUND(overall_min_accuracy, 3),
      'max_accuracy', ROUND(overall_max_accuracy, 3),
      'high_performing_pct', ROUND(high_performing_sessions * 100.0 / (high_performing_sessions + moderate_performing_sessions + low_performing_sessions), 1),
      'sufficient_data_pct', ROUND(sufficient_data_sessions * 100.0 / (sufficient_data_sessions + limited_data_sessions + insufficient_data_sessions), 1)
    ) as metrics_json,
    
    -- Display values
    total_sessions as display_value_1,
    ROUND(overall_avg_accuracy, 3) as display_value_2,
    ROUND(high_performing_sessions * 100.0 / (high_performing_sessions + moderate_performing_sessions + low_performing_sessions), 1) as display_value_3
    
  FROM overall_performance_summary
  
  UNION ALL
  
  SELECT
    'JUDGE_VS_SYNTH' as summary_type,
    metric_category,
    
    JSON_OBJECT(
      'judge_sessions', judge_sessions,
      'synth_sessions', synth_sessions,
      'judge_accuracy', ROUND(judge_avg_accuracy, 3),
      'synth_accuracy', ROUND(synth_avg_accuracy, 3),
      'accuracy_lift', ROUND(simple_accuracy_lift, 3),
      'lift_percentage', ROUND(simple_lift_percentage, 1)
    ) as metrics_json,
    
    ROUND(simple_accuracy_lift, 3) as display_value_1,
    ROUND(simple_lift_percentage, 1) as display_value_2,
    judge_sessions + synth_sessions as display_value_3
    
  FROM judge_vs_synth_summary
  
  UNION ALL
  
  SELECT
    'SYNTH_LIFT_DETAIL' as summary_type,
    metric_category,
    
    JSON_OBJECT(
      'total_lift_sessions', total_lift_sessions,
      'positive_lift_pct', ROUND(positive_lift_sessions * 100.0 / total_lift_sessions, 1),
      'avg_lift', ROUND(avg_absolute_lift, 3),
      'avg_lift_pct', ROUND(avg_relative_lift_pct, 1),
      'high_value_pct', ROUND(high_value_sessions * 100.0 / total_lift_sessions, 1),
      'large_effect_pct', ROUND(large_effect_sessions * 100.0 / total_lift_sessions, 1)
    ) as metrics_json,
    
    ROUND(positive_lift_sessions * 100.0 / total_lift_sessions, 1) as display_value_1,
    ROUND(avg_absolute_lift, 3) as display_value_2,
    total_lift_sessions as display_value_3
    
  FROM synth_lift_summary
)

SELECT 
  summary_type,
  metric_category,
  metrics_json,
  display_value_1,
  display_value_2,
  display_value_3,
  CURRENT_TIMESTAMP() as summary_generated_at

FROM final_summary

-- Add model and criteria summaries as separate rows for flexible querying
UNION ALL

SELECT
  'MODEL_RANKING' as summary_type,
  CONCAT('Model: ', agent_model) as metric_category,
  JSON_OBJECT(
    'model', agent_model,
    'sessions', model_sessions,
    'accuracy', ROUND(model_avg_accuracy, 3),
    'judge_accuracy', ROUND(model_judge_accuracy, 3),
    'synth_accuracy', ROUND(model_synth_accuracy, 3),
    'rank', model_accuracy_rank
  ) as metrics_json,
  model_accuracy_rank as display_value_1,
  ROUND(model_avg_accuracy, 3) as display_value_2,
  model_sessions as display_value_3,
  CURRENT_TIMESTAMP() as summary_generated_at
FROM model_performance_summary

UNION ALL

SELECT
  'CRITERIA_RANKING' as summary_type,
  CONCAT('Criteria: ', criteria) as metric_category,
  JSON_OBJECT(
    'criteria', criteria,
    'sessions', criteria_sessions,
    'accuracy', ROUND(criteria_avg_accuracy, 3),
    'judge_accuracy', ROUND(criteria_judge_accuracy, 3),
    'synth_accuracy', ROUND(criteria_synth_accuracy, 3),
    'rank', criteria_accuracy_rank
  ) as metrics_json,
  criteria_accuracy_rank as display_value_1,
  ROUND(criteria_avg_accuracy, 3) as display_value_2,
  criteria_sessions as display_value_3,
  CURRENT_TIMESTAMP() as summary_generated_at
FROM criteria_performance_summary

ORDER BY summary_type, display_value_1