-- Test that the evaluation pipeline is complete for active experiments
-- Ensures FETCH -> JUDGE -> SYNTH -> SCORER -> DIFF flow completion

WITH pipeline_completeness AS (
  SELECT 
    session_id,
    record_id,
    judge_criteria,
    
    -- Count each type of agent in the session
    COUNT(DISTINCT CASE WHEN agent_role = 'FETCH' THEN call_id END) as fetch_count,
    COUNT(DISTINCT CASE WHEN agent_role = 'JUDGE' THEN call_id END) as judge_count,
    COUNT(DISTINCT CASE WHEN agent_role = 'SYNTHESISER' THEN call_id END) as synth_count,
    COUNT(DISTINCT CASE WHEN agent_role = 'SCORERS' THEN call_id END) as scorer_count,
    
    -- Check for presence of each pipeline stage
    COUNT(DISTINCT CASE WHEN agent_role = 'FETCH' THEN call_id END) > 0 as has_fetch,
    COUNT(DISTINCT CASE WHEN agent_role = 'JUDGE' THEN call_id END) > 0 as has_judge,
    COUNT(DISTINCT CASE WHEN agent_role = 'SYNTHESISER' THEN call_id END) > 0 as has_synth,
    COUNT(DISTINCT CASE WHEN agent_role = 'SCORERS' THEN call_id END) > 0 as has_scorer,
    
    -- Overall completeness assessment
    CASE 
      WHEN COUNT(DISTINCT CASE WHEN agent_role = 'FETCH' THEN call_id END) > 0
       AND COUNT(DISTINCT CASE WHEN agent_role = 'JUDGE' THEN call_id END) > 0
       AND COUNT(DISTINCT CASE WHEN agent_role = 'SCORERS' THEN call_id END) > 0
      THEN 'complete'
      WHEN COUNT(DISTINCT CASE WHEN agent_role = 'JUDGE' THEN call_id END) > 0
      THEN 'partial'
      ELSE 'failed'
    END as pipeline_status
    
  FROM {{ ref('stg_flows') }}
  WHERE timestamp >= '{{ var("cutoff_date") }}'
  GROUP BY 1, 2, 3
),

pipeline_issues AS (
  SELECT 
    session_id,
    record_id,
    judge_criteria,
    pipeline_status,
    
    CASE 
      WHEN NOT has_fetch THEN 'Missing FETCH stage'
      WHEN NOT has_judge THEN 'Missing JUDGE stage'  
      WHEN NOT has_scorer AND has_judge THEN 'JUDGE predictions without SCORER evaluation'
      WHEN judge_count > 5 AND NOT has_synth THEN 'Multiple JUDGE predictions without SYNTHESISER'
      ELSE NULL
    END as pipeline_issue,
    
    CONCAT(
      'FETCH:', fetch_count, 
      ' JUDGE:', judge_count,
      ' SYNTH:', synth_count, 
      ' SCORER:', scorer_count
    ) as stage_counts
    
  FROM pipeline_completeness
  WHERE pipeline_status != 'complete'
),

session_summary AS (
  SELECT 
    judge_criteria,
    pipeline_status,
    COUNT(*) as session_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY judge_criteria), 1) as percentage
  FROM pipeline_completeness
  GROUP BY 1, 2
)

-- Return sessions with pipeline problems
SELECT 
  session_id,
  record_id,
  judge_criteria,
  pipeline_issue as description,
  stage_counts as details
FROM pipeline_issues
WHERE pipeline_issue IS NOT NULL

-- Also include summary if completion rate is concerning  
UNION ALL

SELECT 
  'SUMMARY' as session_id,
  judge_criteria as record_id,
  CONCAT(pipeline_status, ' sessions') as judge_criteria,
  CONCAT('Completion rate concern: ', percentage, '% ', pipeline_status) as description,
  CONCAT(session_count, ' sessions') as details
FROM session_summary
WHERE 
  (pipeline_status = 'failed' AND percentage > 10)  -- More than 10% failed
  OR (pipeline_status = 'partial' AND percentage > 25)  -- More than 25% partial

ORDER BY session_id, record_id