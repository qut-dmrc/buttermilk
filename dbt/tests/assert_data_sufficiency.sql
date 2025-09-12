-- Test to ensure we have sufficient data for meaningful analysis
-- This test WARNS when data sufficiency is questionable but doesn't fail the build

WITH data_summary AS (
  SELECT 
    judge_criteria,
    judge_model,
    COUNT(DISTINCT record_id) as records_evaluated,
    COUNT(DISTINCT CASE WHEN score_quality = 'valid' THEN call_id END) as valid_predictions,
    COUNT(DISTINCT call_id) as total_predictions,
    
    -- Calculate coverage metrics
    SAFE_DIVIDE(
      COUNT(DISTINCT CASE WHEN score_quality = 'valid' THEN call_id END),
      COUNT(DISTINCT call_id)
    ) as score_coverage,
    
    -- Count runs per record for stochastic validation
    AVG(predictions_per_record) as avg_runs_per_record
    
  FROM {{ ref('judge_scores') }}
  LEFT JOIN (
    SELECT 
      record_id,
      judge_model,
      judge_criteria,
      COUNT(*) as predictions_per_record
    FROM {{ ref('judge_scores') }}
    GROUP BY 1, 2, 3
  ) AS run_counts USING (record_id, judge_model, judge_criteria)
  
  GROUP BY 1, 2
),

insufficient_data AS (
  SELECT 
    judge_criteria,
    judge_model,
    records_evaluated,
    valid_predictions,
    score_coverage,
    avg_runs_per_record,
    
    -- Define what constitutes insufficient data
    CASE 
      WHEN records_evaluated < 5 THEN 'Too few records evaluated'
      WHEN score_coverage < {{ var('min_coverage_threshold') }} THEN 'Score coverage below threshold'
      WHEN avg_runs_per_record < {{ var('min_stochastic_runs') }} THEN 'Insufficient runs for stochastic testing'
      ELSE NULL
    END as insufficiency_reason
    
  FROM data_summary
  WHERE 
    records_evaluated < 5
    OR score_coverage < {{ var('min_coverage_threshold') }}
    OR avg_runs_per_record < {{ var('min_stochastic_runs') }}
)

-- Return rows that indicate data sufficiency problems
SELECT 
  judge_criteria,
  judge_model,
  insufficiency_reason,
  records_evaluated,
  ROUND(score_coverage, 3) as score_coverage,
  ROUND(avg_runs_per_record, 1) as avg_runs_per_record
FROM insufficient_data
ORDER BY judge_criteria, judge_model