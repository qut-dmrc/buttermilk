-- Test experimental integrity: ensure proper A/B test structure and golden set coverage
-- This validates that experiments are set up correctly for meaningful analysis

WITH experiment_structure AS (
  SELECT 
    judge_criteria,
    record_id,
    COUNT(DISTINCT judge_template) as template_variants,
    COUNT(DISTINCT judge_model) as model_variants,
    COUNT(*) as total_predictions,
    
    -- Check for balanced A/B testing
    MIN(template_count) as min_template_usage,
    MAX(template_count) as max_template_usage,
    
    -- Calculate experiment hash for grouping
    {{ get_experiment_hash('judge_model', 'judge_criteria', 'judge_template', 'record_id') }} as experiment_hash
    
  FROM {{ ref('judge_scores') }}
  LEFT JOIN (
    SELECT 
      judge_template,
      judge_criteria,
      record_id,
      COUNT(*) as template_count
    FROM {{ ref('judge_scores') }}
    GROUP BY 1, 2, 3
  ) template_usage USING (judge_template, judge_criteria, record_id)
  
  WHERE score_quality = 'valid'  -- Only consider valid predictions
  GROUP BY 1, 2, experiment_hash
),

integrity_issues AS (
  SELECT 
    judge_criteria,
    record_id,
    experiment_hash,
    template_variants,
    model_variants, 
    total_predictions,
    
    CASE 
      WHEN template_variants < 2 AND total_predictions > 5 THEN 'Missing A/B template variants'
      WHEN model_variants < 2 AND total_predictions > 10 THEN 'Insufficient model diversity'
      WHEN total_predictions < {{ var('min_stochastic_runs') }} THEN 'Insufficient runs for stochastic testing'
      WHEN ABS(max_template_usage - min_template_usage) > (max_template_usage * 0.5) THEN 'Unbalanced A/B testing'
      ELSE NULL
    END as integrity_issue
    
  FROM experiment_structure
  WHERE 
    -- Flag experiments with structural problems
    template_variants < 2 AND total_predictions > 5
    OR model_variants < 2 AND total_predictions > 10  
    OR total_predictions < {{ var('min_stochastic_runs') }}
    OR ABS(max_template_usage - min_template_usage) > (max_template_usage * 0.5)
),

golden_set_coverage AS (
  SELECT 
    judge_criteria,
    COUNT(DISTINCT record_id) as records_covered,
    
    -- Estimate expected golden set size (this would be configurable)
    CASE judge_criteria
      WHEN 'TJA' THEN 28  -- Known from eval.md
      ELSE 20  -- Default assumption
    END as expected_records,
    
    -- Calculate coverage percentage
    SAFE_DIVIDE(
      COUNT(DISTINCT record_id),
      CASE judge_criteria WHEN 'TJA' THEN 28 ELSE 20 END
    ) as coverage_rate
    
  FROM {{ ref('judge_scores') }}
  WHERE score_quality = 'valid'
  GROUP BY 1
)

-- Return all integrity issues found
SELECT 
  'Experiment Structure' as issue_type,
  judge_criteria,
  CAST(record_id AS STRING) as affected_item,
  integrity_issue as description,
  CONCAT('Templates: ', template_variants, ', Models: ', model_variants, ', Predictions: ', total_predictions) as details
FROM integrity_issues

UNION ALL

SELECT 
  'Golden Set Coverage' as issue_type,
  judge_criteria,
  'Overall' as affected_item,
  CONCAT('Coverage below threshold: ', ROUND(coverage_rate * 100, 1), '%') as description,
  CONCAT('Covered: ', records_covered, '/', expected_records, ' records') as details
FROM golden_set_coverage 
WHERE coverage_rate < {{ var('min_coverage_threshold') }}

ORDER BY issue_type, judge_criteria