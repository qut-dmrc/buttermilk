{{ config(
    materialized='table',
    description='Multi-dimensional analysis for template A/B testing across all combinations'
) }}

-- Create aggregated views for different dimensional cuts
WITH base_performance AS (
  SELECT
    template_label,
    model,
    criteria,
    agent_role,
    avg_accuracy,
    total_predictions,
    data_sufficiency,
    absolute_lift,
    relative_lift_percent
  FROM {{ ref('template_performance_comparison') }}
),

-- 1. Overall template performance (across all dimensions)
overall_template_performance AS (
  SELECT
    'OVERALL' as dimension_type,
    template_label as dimension_value,
    '' as secondary_dimension,
    AVG(avg_accuracy) as avg_accuracy,
    SUM(total_predictions) as total_predictions,
    STDDEV(avg_accuracy) as accuracy_variance,
    COUNT(*) as dimension_combinations
  FROM base_performance
  WHERE data_sufficiency = 'SUFFICIENT'
  GROUP BY template_label
),

-- 2. Performance by model (across templates)
model_performance AS (
  SELECT
    'MODEL' as dimension_type,
    model as dimension_value,
    template_label as secondary_dimension,
    AVG(avg_accuracy) as avg_accuracy,
    SUM(total_predictions) as total_predictions,
    STDDEV(avg_accuracy) as accuracy_variance,
    COUNT(*) as dimension_combinations
  FROM base_performance
  WHERE data_sufficiency = 'SUFFICIENT'
  GROUP BY model, template_label
),

-- 3. Performance by criteria (across templates)
criteria_performance AS (
  SELECT
    'CRITERIA' as dimension_type,
    criteria as dimension_value,
    template_label as secondary_dimension,
    AVG(avg_accuracy) as avg_accuracy,
    SUM(total_predictions) as total_predictions,
    STDDEV(avg_accuracy) as accuracy_variance,
    COUNT(*) as dimension_combinations
  FROM base_performance
  WHERE data_sufficiency = 'SUFFICIENT'
  GROUP BY criteria, template_label
),

-- 4. Performance by agent type (across templates)
agent_performance AS (
  SELECT
    'AGENT_TYPE' as dimension_type,
    agent_role as dimension_value,
    template_label as secondary_dimension,
    AVG(avg_accuracy) as avg_accuracy,
    SUM(total_predictions) as total_predictions,
    STDDEV(avg_accuracy) as accuracy_variance,
    COUNT(*) as dimension_combinations
  FROM base_performance
  WHERE data_sufficiency = 'SUFFICIENT'
  GROUP BY agent_role, template_label
),

-- 5. Model-Criteria interaction effects
model_criteria_interaction AS (
  SELECT
    'MODEL_CRITERIA' as dimension_type,
    CONCAT(model, ' + ', criteria) as dimension_value,
    template_label as secondary_dimension,
    AVG(avg_accuracy) as avg_accuracy,
    SUM(total_predictions) as total_predictions,
    STDDEV(avg_accuracy) as accuracy_variance,
    COUNT(*) as dimension_combinations
  FROM base_performance
  WHERE data_sufficiency = 'SUFFICIENT'
  GROUP BY model, criteria, template_label
),

-- Combine all dimensional views
all_dimensions AS (
  SELECT * FROM overall_template_performance
  UNION ALL SELECT * FROM model_performance
  UNION ALL SELECT * FROM criteria_performance  
  UNION ALL SELECT * FROM agent_performance
  UNION ALL SELECT * FROM model_criteria_interaction
),

-- Calculate lift for each dimensional cut
dimensional_lift AS (
  SELECT
    dimension_type,
    dimension_value,
    
    -- Template A metrics
    MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END) as template_a_accuracy,
    MAX(CASE WHEN secondary_dimension = 'Template A' THEN total_predictions END) as template_a_predictions,
    
    -- Template B metrics
    MAX(CASE WHEN secondary_dimension = 'Template B' THEN avg_accuracy END) as template_b_accuracy,
    MAX(CASE WHEN secondary_dimension = 'Template B' THEN total_predictions END) as template_b_predictions,
    
    -- Calculate lift
    (MAX(CASE WHEN secondary_dimension = 'Template B' THEN avg_accuracy END) - 
     MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END)) as absolute_lift,
    
    CASE 
      WHEN MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END) > 0 THEN
        ((MAX(CASE WHEN secondary_dimension = 'Template B' THEN avg_accuracy END) - 
          MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END)) / 
         MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END)) * 100
      ELSE NULL
    END as relative_lift_percent,
    
    -- Statistical significance indicators
    CASE 
      WHEN MAX(CASE WHEN secondary_dimension = 'Template A' THEN total_predictions END) >= 30 
       AND MAX(CASE WHEN secondary_dimension = 'Template B' THEN total_predictions END) >= 30 
      THEN 'HIGH_CONFIDENCE'
      WHEN MAX(CASE WHEN secondary_dimension = 'Template A' THEN total_predictions END) >= 10 
       AND MAX(CASE WHEN secondary_dimension = 'Template B' THEN total_predictions END) >= 10 
      THEN 'MEDIUM_CONFIDENCE'
      ELSE 'LOW_CONFIDENCE'
    END as statistical_confidence,
    
    -- Effect size classification
    CASE 
      WHEN ABS(MAX(CASE WHEN secondary_dimension = 'Template B' THEN avg_accuracy END) - 
               MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END)) >= 0.1 
      THEN 'LARGE_EFFECT'
      WHEN ABS(MAX(CASE WHEN secondary_dimension = 'Template B' THEN avg_accuracy END) - 
               MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END)) >= 0.05 
      THEN 'MEDIUM_EFFECT'
      WHEN ABS(MAX(CASE WHEN secondary_dimension = 'Template B' THEN avg_accuracy END) - 
               MAX(CASE WHEN secondary_dimension = 'Template A' THEN avg_accuracy END)) >= 0.01 
      THEN 'SMALL_EFFECT'
      ELSE 'NEGLIGIBLE_EFFECT'
    END as effect_size
    
  FROM all_dimensions
  WHERE secondary_dimension IN ('Template A', 'Template B')
  GROUP BY dimension_type, dimension_value
  HAVING COUNT(DISTINCT secondary_dimension) = 2 -- Ensure both templates present
)

-- Final output with rankings and insights
SELECT
  dimension_type,
  dimension_value,
  template_a_accuracy,
  template_a_predictions,
  template_b_accuracy,
  template_b_predictions,
  absolute_lift,
  relative_lift_percent,
  statistical_confidence,
  effect_size,
  
  -- Rankings by lift magnitude
  ROW_NUMBER() OVER (
    PARTITION BY dimension_type 
    ORDER BY ABS(relative_lift_percent) DESC
  ) as lift_magnitude_rank,
  
  -- Direction indicators
  CASE 
    WHEN absolute_lift > 0 THEN 'Template B Better'
    WHEN absolute_lift < 0 THEN 'Template A Better'
    ELSE 'No Difference'
  END as performance_direction,
  
  -- Recommendation flags
  CASE 
    WHEN statistical_confidence = 'HIGH_CONFIDENCE' 
     AND effect_size IN ('LARGE_EFFECT', 'MEDIUM_EFFECT')
     AND absolute_lift > 0 
    THEN 'STRONG_RECOMMEND_B'
    WHEN statistical_confidence = 'HIGH_CONFIDENCE' 
     AND effect_size IN ('LARGE_EFFECT', 'MEDIUM_EFFECT')
     AND absolute_lift < 0 
    THEN 'STRONG_RECOMMEND_A'
    WHEN statistical_confidence IN ('HIGH_CONFIDENCE', 'MEDIUM_CONFIDENCE') 
     AND effect_size = 'SMALL_EFFECT'
    THEN 'WEAK_RECOMMENDATION'
    ELSE 'INSUFFICIENT_EVIDENCE'
  END as recommendation
  
FROM dimensional_lift
ORDER BY 
  CASE dimension_type 
    WHEN 'OVERALL' THEN 1
    WHEN 'MODEL' THEN 2  
    WHEN 'CRITERIA' THEN 3
    WHEN 'AGENT_TYPE' THEN 4
    WHEN 'MODEL_CRITERIA' THEN 5
  END,
  ABS(relative_lift_percent) DESC