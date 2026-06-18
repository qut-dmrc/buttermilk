-- NOTE (task buttermilk-1e23cce6): the source record is now read from the canonical
-- top-level `record` column (a RECORD/STRUCT), not from a copy embedded in `inputs`
-- (`inputs.$.records` is no longer emitted — canonical-once). Re-apply this DDL manually
-- and VALIDATE against the live table; the trace schema is traces.schema.json.
CREATE VIEW `{DATASET}.judge_reasons`
AS
  SELECT
    session_id,
    call_id,
    timestamp,
    error,
    record,
    JSON_VALUE(run_info, "$.name") AS name,
    JSON_VALUE(run_info, "$.job") AS job,
    record.record_id AS record_id,
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
  WHERE
    timestamp >= DATETIME_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    AND JSON_VALUE(agent_info, "$.role") IN ('JUDGE', 'SYNTHESISER')
ORDER BY timestamp DESC
