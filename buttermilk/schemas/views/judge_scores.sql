-- NOTE (task buttermilk-1e23cce6): record + ground_truth are now read from the canonical
-- top-level `record` column (a RECORD/STRUCT: record.record_id, record.ground_truth), not
-- from a copy in `inputs.$.records` (no longer emitted). Re-apply manually and VALIDATE
-- against the live table; trace schema is traces.schema.json (record.ground_truth is JSON).
WITH SCORES_AGGREGATED AS (
  SELECT
    session_id,
    call_id AS scorer_call_id,
    timestamp,
    error,
    tracing_link,
    IFNULL(parent_call_id, JSON_VALUE(inputs, "$.inputs.answers[0].answer_id")) as parent_call_id,
    JSON_VALUE(agent_info, "$.name") AS scorer,
    JSON_VALUE(agent_info, "$.parameters.model") AS scorer_model,
    JSON_VALUE(metadata, "$.template_name") AS scorer_template,
    JSON_VALUE(metadata, "$.template_hash") AS scorer_hash,
    JSON_VALUE(agent_info, "$.role") AS role,
    CAST(JSON_VALUE(outputs, "$.correctness") AS FLOAT64) AS correctness,
    JSON_EXTRACT_ARRAY(outputs, '$.assessments') AS assessments
  FROM
    `prosocial-443205.testing.flow`
  WHERE
    JSON_VALUE(agent_info, "$.role") IN ('SCORERS')
),
PREDICTIONS AS (
  SELECT
    session_id,
    call_id,
    timestamp,
    error,
    record,
    tracing_link,
    record.record_id AS record_id,
    JSON_VALUE(agent_info, "$.name") AS judge,
    JSON_VALUE(agent_info, "$.parameters.model") AS judge_model,
    JSON_VALUE(metadata, "$.template_name") AS judge_template,
    JSON_VALUE(metadata, "$.template_hash") AS judge_hash,
    JSON_VALUE(agent_info, "$.parameters.criteria") AS judge_criteria,
    JSON_VALUE(agent_info, "$.role") AS judge_role,
    -- Predicted Label
    CAST(JSON_VALUE(outputs, "$.prediction") AS BOOLEAN) AS predicted_violating,
    -- True Label (from the canonical record column; record.ground_truth is JSON)
    CAST(JSON_VALUE(record.ground_truth, '$.violating') AS BOOLEAN) AS expected_violating,
    -- Calculate full_prediction_summary within this CTE
    CONCAT(
      IFNULL(JSON_VALUE(outputs, "$.conclusion"), ''),
      ' ', -- Add a space separator
      ARRAY_TO_STRING(JSON_EXTRACT_STRING_ARRAY(outputs, "$.reasons"), '. ', '') -- Join reasons with '. ' and an empty string for nulls
    ) AS full_reasons,
    JSON_VALUE(outputs, "$.confidence") AS confidence
  FROM
    `prosocial-443205.testing.flow`
  WHERE
    JSON_VALUE(agent_info, "$.role") IN ('JUDGE', 'SYNTHESISER')
)
SELECT
  PREDICTIONS.session_id,
  PREDICTIONS.call_id,
  SCORES_AGGREGATED.scorer_call_id,
  PREDICTIONS.timestamp,
  ARRAY_AGG(PREDICTIONS.record IGNORE NULLS) AS record,
  PREDICTIONS.record_id,
  PREDICTIONS.judge,
  PREDICTIONS.judge_model,
  PREDICTIONS.judge_template,
  SUBSTR(PREDICTIONS.judge_hash, 8, 8) as judge_hash, -- remove 'SHA256:' prefix and return first eight digits of hash (4 billion unique ids)
  PREDICTIONS.judge_criteria,
  PREDICTIONS.judge_role,
  PREDICTIONS.full_reasons, -- Use the pre-calculated summary
  PREDICTIONS.confidence,
  PREDICTIONS.tracing_link,
  SCORES_AGGREGATED.scorer,
  SCORES_AGGREGATED.scorer_model,
  SCORES_AGGREGATED.scorer_template,
  SUBSTR(SCORES_AGGREGATED.scorer_hash, 8, 8) as scorer_hash,
  SCORES_AGGREGATED.role,
  SCORES_AGGREGATED.tracing_link as scorer_tracing_link,
  predicted_violating,
  expected_violating,
  CAST((PREDICTIONS.predicted_violating = PREDICTIONS.expected_violating) AS BOOLEAN) as correct,
  SCORES_AGGREGATED.correctness,
  ARRAY_AGG(CAST(JSON_VALUE(assessment, '$.correct') AS BOOLEAN) IGNORE NULLS) AS assessment_correct,
  ARRAY_AGG(JSON_VALUE(assessment, '$.feedback') IGNORE NULLS) AS assessment_feedback,
FROM
  PREDICTIONS
LEFT JOIN
  SCORES_AGGREGATED ON PREDICTIONS.call_id = SCORES_AGGREGATED.parent_call_id
LEFT JOIN
  UNNEST(SCORES_AGGREGATED.assessments) AS assessment
WHERE
  TRUE
  -- Bad data before this date
  AND PREDICTIONS.timestamp >= '2025-05-01' AND SCORES_AGGREGATED.timestamp >= '2025-05-01'

  -- This should make the join somewhat faster
  AND SCORES_AGGREGATED.timestamp >= PREDICTIONS.timestamp

  -- Uncomment these for a snappy last week
--  AND PREDICTIONS.timestamp >= DATETIME_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
--  AND SCORES_AGGREGATED.timestamp >= DATETIME_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY
  PREDICTIONS.session_id,
  PREDICTIONS.call_id,
  SCORES_AGGREGATED.scorer_call_id,
  PREDICTIONS.timestamp,
  PREDICTIONS.record_id,
  PREDICTIONS.judge,
  PREDICTIONS.judge_model,
  PREDICTIONS.judge_template,
  PREDICTIONS.judge_hash,
  PREDICTIONS.judge_criteria,
  PREDICTIONS.judge_role,
  PREDICTIONS.full_reasons,
  PREDICTIONS.confidence,
  PREDICTIONS.tracing_link,
  SCORES_AGGREGATED.tracing_link ,
  SCORES_AGGREGATED.scorer,
  SCORES_AGGREGATED.scorer_model,
  SCORES_AGGREGATED.scorer_template,
  SCORES_AGGREGATED.scorer_hash,
  SCORES_AGGREGATED.role,
  SCORES_AGGREGATED.correctness,
  predicted_violating,
  expected_violating
ORDER BY
  PREDICTIONS.timestamp DESC;
