CREATE VIEW `{DATASET}.judge_reasons`
AS
  SELECT
    session_id,
    call_id,
    timestamp,
    error,
    records as record,
    JSON_VALUE(run_info, "$.name") AS name,
    JSON_VALUE(run_info, "$.job") AS job,
    JSON_VALUE(records, '$.record_id') AS record_id,
    JSON_VALUE(agent_info, "$.name") AS judge,
    JSON_VALUE(agent_info, "$.parameters.model") AS judge_model,
    JSON_VALUE(agent_info, "$.parameters.template") AS judge_template,
    JSON_VALUE(agent_info, "$.parameters.criteria") AS judge_criteria,
    JSON_VALUE(agent_info, "$.role") AS judge_role,
    JSON_EXTRACT_STRING_ARRAY(outputs, "$.reasons") AS reasons,
    JSON_VALUE(outputs, "$.conclusion") AS conclusion,
    CAST(JSON_VALUE(outputs, "$.prediction") AS BOOLEAN) AS violating,
    JSON_VALUE(outputs, "$.confidence") AS confidence,
    tracing_link, parent_call_id
  FROM
    `{DATASET}.{FLOWS_TABLE}`
    LEFT JOIN UNNEST(JSON_QUERY_ARRAY(inputs, '$.records')) AS records
  WHERE
    timestamp >= DATETIME_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    AND JSON_VALUE(agent_info, "$.role") IN ('JUDGE', 'SYNTHESISER')
ORDER BY timestamp DESC