WITH SCORES_AGGREGATED AS (
  SELECT
    session_id,
    call_id AS scorer_call_id,
    timestamp,
    error,
    IFNULL(parent_call_id, JSON_VALUE(inputs, "$.inputs.answers[0].answer_id")) as parent_call_id,
    JSON_VALUE(agent_info, "$.name") AS scorer,
    JSON_VALUE(agent_info, "$.parameters.model") AS scoring_model,
    JSON_VALUE(agent_info, "$.parameters.template") AS scoring_template,
    JSON_VALUE(agent_info, "$.role") AS role,
    CAST(JSON_VALUE(outputs, "$.correctness") AS FLOAT64) AS correctness,
    JSON_EXTRACT_ARRAY(outputs, '$.assessments') AS assessments
  FROM
    `{DATASET}.{FLOWS_TABLE}`
  WHERE
    JSON_VALUE(agent_info, "$.role") IN ('SCORERS')
),
PREDICTIONS AS (
  SELECT
    session_id,
    call_id,
    timestamp,
    error,
    records as record,
    JSON_VALUE(records, '$.record_id') AS record_id,
    JSON_VALUE(agent_info, "$.name") AS judge,
    JSON_VALUE(agent_info, "$.parameters.model") AS judge_model,
    JSON_VALUE(agent_info, "$.parameters.template") AS judge_template,
    JSON_VALUE(agent_info, "$.parameters.criteria") AS judge_criteria,
    JSON_VALUE(agent_info, "$.role") AS judge_role,
    JSON_EXTRACT_STRING_ARRAY(outputs, "$.reasons") AS reasons,
    JSON_VALUE(outputs, "$.conclusion") AS conclusion,
    CAST(JSON_VALUE(outputs, "$.prediction") AS BOOLEAN) AS violating,
    JSON_VALUE(outputs, "$.confidence") AS confidence
  FROM
    `{DATASET}.{FLOWS_TABLE}`
    LEFT JOIN UNNEST(JSON_QUERY_ARRAY(inputs, '$.records')) AS records
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
  PREDICTIONS.judge_criteria,
  PREDICTIONS.judge_role,
  PREDICTIONS.reasons,
  PREDICTIONS.conclusion,
  PREDICTIONS.violating,
  PREDICTIONS.confidence,
  SCORES_AGGREGATED.scorer,
  SCORES_AGGREGATED.scoring_model,
  SCORES_AGGREGATED.scoring_template,
  SCORES_AGGREGATED.role,
  ARRAY_AGG(CAST(JSON_VALUE(assessment, '$.correct') AS BOOLEAN) IGNORE NULLS) AS assessment_correct,
  ARRAY_AGG(JSON_VALUE(assessment, '$.feedback') IGNORE NULLS) AS assessment_feedback,
  SCORES_AGGREGATED.correctness
FROM
  PREDICTIONS
LEFT JOIN
  SCORES_AGGREGATED ON PREDICTIONS.call_id = SCORES_AGGREGATED.parent_call_id
LEFT JOIN
  UNNEST(SCORES_AGGREGATED.assessments) AS assessment
WHERE
  PREDICTIONS.timestamp >= DATETIME_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND SCORES_AGGREGATED.timestamp >= DATETIME_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY
  PREDICTIONS.session_id,
  PREDICTIONS.call_id,
  SCORES_AGGREGATED.scorer_call_id,
  PREDICTIONS.timestamp,
  PREDICTIONS.record_id,
  PREDICTIONS.judge,
  PREDICTIONS.judge_model,
  PREDICTIONS.judge_template,
  PREDICTIONS.judge_criteria,
  PREDICTIONS.judge_role,
  PREDICTIONS.reasons,
  PREDICTIONS.conclusion,
  PREDICTIONS.violating,
  PREDICTIONS.confidence,
  SCORES_AGGREGATED.scorer,
  SCORES_AGGREGATED.scoring_model,
  SCORES_AGGREGATED.scoring_template,
  SCORES_AGGREGATED.role,
  SCORES_AGGREGATED.correctness 
ORDER BY
  PREDICTIONS.timestamp DESC;