{{ config(
    materialized='table',
    description='Multi-dimensional performance matrix showing all permutations of Model × Criteria × Experiment × Agent Type for dashboard pivot tables'
) }}

-- Performance matrix for comprehensive dimensional analysis
-- Optimized for dashboard consumption with pivot tables and heatmaps
-- Shows performance across all combinations of key dimensions

WITH base_performance AS (
  SELECT
    -- Core dimensions
    agent_model,
    criteria,
    agent_role,
    experiment_name,
    experiment_label,
    experiment_date,
    
    -- Performance aggregations
    AVG(avg_accuracy) as avg_accuracy,
    AVG(binary_accuracy) as avg_binary_accuracy,
    AVG(avg_confidence) as avg_confidence,
    
    -- Performance distribution
    STDDEV(avg_accuracy) as accuracy_stddev,
    MIN(avg_accuracy) as min_accuracy,
    MAX(avg_accuracy) as max_accuracy,
    APPROX_QUANTILES(avg_accuracy, 4)[OFFSET(1)] as q25_accuracy,
    APPROX_QUANTILES(avg_accuracy, 4)[OFFSET(2)] as median_accuracy,
    APPROX_QUANTILES(avg_accuracy, 4)[OFFSET(3)] as q75_accuracy,
    
    -- Volume and data quality
    COUNT(*) as session_count,
    SUM(prediction_count) as total_predictions,
    SUM(total_scorer_evaluations) as total_scorer_evaluations,
    
    -- Data sufficiency assessment
    COUNT(CASE WHEN data_sufficiency = 'SUFFICIENT' THEN 1 END) as sufficient_sessions,
    COUNT(CASE WHEN data_sufficiency = 'LIMITED' THEN 1 END) as limited_sessions,
    COUNT(CASE WHEN data_sufficiency = 'INSUFFICIENT' THEN 1 END) as insufficient_sessions,
    
    -- Performance quality
    COUNT(CASE WHEN performance_category = 'HIGH_PERFORMING' THEN 1 END) as high_performing_sessions,
    COUNT(CASE WHEN performance_category = 'MODERATE_PERFORMING' THEN 1 END) as moderate_performing_sessions,
    COUNT(CASE WHEN performance_category = 'LOW_PERFORMING' THEN 1 END) as low_performing_sessions,
    
    -- Timing
    MIN(first_prediction) as earliest_prediction,
    MAX(last_prediction) as latest_prediction
    
  FROM {{ ref('individual_agent_performance') }}
  WHERE data_sufficiency IN ('SUFFICIENT', 'LIMITED')  -- Focus on quality data
  GROUP BY 1, 2, 3, 4, 5, 6
),

-- Model × Criteria performance matrix
model_criteria_matrix AS (
  SELECT
    'Model_Criteria' as dimension_type,
    agent_model as dimension_1,
    criteria as dimension_2,
    agent_role as dimension_3,
    experiment_name as dimension_4,
    
    avg_accuracy,
    avg_binary_accuracy,
    avg_confidence,
    accuracy_stddev,
    session_count,
    total_predictions,
    
    -- Performance ranking within this dimension
    RANK() OVER (PARTITION BY agent_role ORDER BY avg_accuracy DESC) as performance_rank,
    PERCENT_RANK() OVER (PARTITION BY agent_role ORDER BY avg_accuracy) as performance_percentile
    
  FROM base_performance
),

-- Model × Agent Type performance matrix
model_agent_matrix AS (
  SELECT
    'Model_Agent' as dimension_type,
    agent_model as dimension_1,
    agent_role as dimension_2,
    criteria as dimension_3,
    experiment_name as dimension_4,
    
    avg_accuracy,
    avg_binary_accuracy,
    avg_confidence,
    accuracy_stddev,
    session_count,
    total_predictions,
    
    RANK() OVER (PARTITION BY criteria ORDER BY avg_accuracy DESC) as performance_rank,
    PERCENT_RANK() OVER (PARTITION BY criteria ORDER BY avg_accuracy) as performance_percentile
    
  FROM base_performance
),

-- Criteria × Agent Type performance matrix
criteria_agent_matrix AS (
  SELECT
    'Criteria_Agent' as dimension_type,
    criteria as dimension_1,
    agent_role as dimension_2,
    agent_model as dimension_3,
    experiment_name as dimension_4,
    
    avg_accuracy,
    avg_binary_accuracy,
    avg_confidence,
    accuracy_stddev,
    session_count,
    total_predictions,
    
    RANK() OVER (PARTITION BY agent_model ORDER BY avg_accuracy DESC) as performance_rank,
    PERCENT_RANK() OVER (PARTITION BY agent_model ORDER BY avg_accuracy) as performance_percentile
    
  FROM base_performance
),

-- Experiment × Agent Type performance matrix
experiment_agent_matrix AS (
  SELECT
    'Experiment_Agent' as dimension_type,
    COALESCE(experiment_name, 'Unlabeled') as dimension_1,
    agent_role as dimension_2,
    agent_model as dimension_3,
    criteria as dimension_4,
    
    avg_accuracy,
    avg_binary_accuracy,
    avg_confidence,
    accuracy_stddev,
    session_count,
    total_predictions,
    
    RANK() OVER (PARTITION BY agent_model, criteria ORDER BY avg_accuracy DESC) as performance_rank,
    PERCENT_RANK() OVER (PARTITION BY agent_model, criteria ORDER BY avg_accuracy) as performance_percentile
    
  FROM base_performance
),

-- Combine all matrices
combined_matrix AS (
  SELECT * FROM model_criteria_matrix
  UNION ALL
  SELECT * FROM model_agent_matrix  
  UNION ALL
  SELECT * FROM criteria_agent_matrix
  UNION ALL
  SELECT * FROM experiment_agent_matrix
),

-- Add additional metrics and categorizations
performance_matrix_final AS (
  SELECT
    *,
    
    -- Performance categories for color coding
    CASE 
      WHEN avg_accuracy >= 0.8 THEN 'Excellent'
      WHEN avg_accuracy >= 0.7 THEN 'Good'
      WHEN avg_accuracy >= 0.6 THEN 'Fair'
      WHEN avg_accuracy >= 0.5 THEN 'Poor'
      ELSE 'Very Poor'
    END as performance_grade,
    
    -- Data reliability indicators
    CASE 
      WHEN session_count >= 20 AND total_predictions >= 100 THEN 'High_Reliability'
      WHEN session_count >= 10 AND total_predictions >= 50 THEN 'Medium_Reliability'
      WHEN session_count >= 5 AND total_predictions >= 20 THEN 'Low_Reliability'
      ELSE 'Insufficient_Data'
    END as data_reliability,
    
    -- Consistency indicators
    CASE 
      WHEN accuracy_stddev <= 0.1 THEN 'Very_Consistent'
      WHEN accuracy_stddev <= 0.15 THEN 'Consistent'
      WHEN accuracy_stddev <= 0.2 THEN 'Moderately_Consistent'
      ELSE 'Inconsistent'
    END as consistency_grade,
    
    -- Comparative performance (within dimension type)
    CASE 
      WHEN performance_percentile >= 0.8 THEN 'Top_Performer'
      WHEN performance_percentile >= 0.6 THEN 'Above_Average'
      WHEN performance_percentile >= 0.4 THEN 'Average'
      WHEN performance_percentile >= 0.2 THEN 'Below_Average'
      ELSE 'Poor_Performer'
    END as relative_performance,
    
    -- Dashboard display labels
    CONCAT(dimension_1, ' × ', dimension_2) as primary_dimension_label,
    CONCAT(dimension_3, ' (', dimension_4, ')') as secondary_dimension_label,
    
    -- Metrics for heatmap visualization
    ROUND(avg_accuracy, 3) as accuracy_display,
    ROUND(accuracy_stddev, 3) as stddev_display,
    FORMAT('%d sessions, %d predictions', session_count, total_predictions) as volume_display
    
  FROM combined_matrix
)

SELECT 
  -- Dimension information
  dimension_type,
  dimension_1,
  dimension_2, 
  dimension_3,
  dimension_4,
  primary_dimension_label,
  secondary_dimension_label,
  
  -- Core performance metrics
  avg_accuracy,
  avg_binary_accuracy,
  avg_confidence,
  accuracy_stddev,
  
  -- Volume metrics
  session_count,
  total_predictions,
  
  -- Quality indicators
  performance_grade,
  consistency_grade,
  data_reliability,
  relative_performance,
  
  -- Rankings
  performance_rank,
  performance_percentile,
  
  -- Display formatting
  accuracy_display,
  stddev_display,
  volume_display

FROM performance_matrix_final
ORDER BY 
  dimension_type,
  performance_rank