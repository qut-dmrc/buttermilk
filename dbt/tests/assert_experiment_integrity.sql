-- Test experimental integrity: ensure proper A/B test structure and golden set coverage
-- This test checks if the experiment_integrity_issues monitoring model has any rows
-- If it does, that indicates experiment integrity problems

SELECT *
FROM {{ ref('experiment_integrity_issues') }}