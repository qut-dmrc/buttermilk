-- Test that the evaluation pipeline is complete for active experiments
-- This test checks if the pipeline_completeness_issues monitoring model has any rows
-- If it does, that indicates pipeline completion problems

SELECT *
FROM {{ ref('pipeline_completeness_issues') }}