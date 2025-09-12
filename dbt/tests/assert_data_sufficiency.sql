-- Test to ensure we have sufficient data for meaningful analysis
-- This test checks if the data_sufficiency_issues monitoring model has any rows
-- If it does, that indicates data sufficiency problems

SELECT *
FROM {{ ref('data_sufficiency_issues') }}