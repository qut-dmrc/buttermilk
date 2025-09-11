-- Test to ensure template analysis models have the expected data structure
-- Updated for multi-experiment template hash-based approach

WITH data_checks AS (
  SELECT 
    (SELECT count(*) FROM {{ ref('int_experiment_completeness') }}) as completeness_count,
    (SELECT count(*) FROM {{ ref('template_performance_comparison') }}) as performance_count,
    (SELECT count(distinct template_hash) FROM {{ ref('template_performance_comparison') }}) as unique_templates,
    (SELECT count(distinct experiment_group) FROM {{ ref('template_performance_comparison') }}) as experiment_groups
)

-- Return failure messages for any failing conditions
SELECT 'No experiment completeness data found' as failure_reason
FROM data_checks
WHERE completeness_count = 0

UNION ALL

SELECT 'No template performance data found' as failure_reason  
FROM data_checks
WHERE performance_count = 0

UNION ALL

SELECT CONCAT('Expected multiple templates, found only ', CAST(unique_templates AS STRING)) as failure_reason
FROM data_checks
WHERE unique_templates < 2

UNION ALL

SELECT CONCAT('Expected multiple experiment groups, found only ', CAST(experiment_groups AS STRING)) as failure_reason
FROM data_checks  
WHERE experiment_groups < 2