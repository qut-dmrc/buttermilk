{{
  config(
    description="Deduplicates multiple SCORER evaluations per prediction, handling graceful failures"
  )
}}

WITH scores_with_rank AS (
  SELECT 
    *,
    -- Rank scorers by timestamp to get the most recent valid score
    ROW_NUMBER() OVER (
      PARTITION BY parent_call_id 
      ORDER BY 
        CASE WHEN correctness IS NOT NULL THEN 0 ELSE 1 END,  -- Prioritize valid scores
        timestamp DESC
    ) as score_rank
  FROM {{ ref('int_scores_aggregated') }}
  WHERE parent_call_id IS NOT NULL  -- Exclude orphaned scores
),

deduped_scores AS (
  SELECT 
    session_id,
    scorer_call_id,
    parent_call_id,
    timestamp,
    scorer,
    scorer_model,
    scorer_template,
    scorer_hash,
    role,
    correctness,
    assessments,
    scorer_tracing_link,
    
    -- Flag for data quality tracking
    CASE 
      WHEN correctness IS NULL THEN 'missing_score'
      WHEN score_rank > 1 THEN 'duplicate_resolved' 
      ELSE 'valid'
    END as score_quality
    
  FROM scores_with_rank
  WHERE score_rank = 1  -- Take the best available score per prediction
)

SELECT * FROM deduped_scores