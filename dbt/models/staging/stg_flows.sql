WITH source AS (
    SELECT * FROM {{ source('testing', 'flow') }}
)
SELECT
    -- IDs and Timestamps
    session_id,
    call_id,
    timestamp,
    IFNULL(parent_call_id, JSON_VALUE(inputs, "$.inputs.answers[0].answer_id")) as parent_call_id,
    JSON_VALUE(records, '$.record_id') AS record_id,

    -- Agent Info
    JSON_VALUE(agent_info, "$.name") AS agent_name,
    JSON_VALUE(agent_info, "$.parameters.model") AS agent_model,
    JSON_VALUE(agent_info, "$.role") AS agent_role,
    JSON_VALUE(agent_info, "$.parameters.criteria") AS judge_criteria,

    -- Metadata
    JSON_VALUE(metadata, "$.template_name") AS template_name,
    JSON_VALUE(metadata, "$.template_hash") AS template_hash,

    -- Inputs and Outputs
    JSON_EXTRACT_ARRAY(outputs, '$.assessments') AS assessments,
    CAST(JSON_VALUE(outputs, "$.correctness") AS FLOAT64) AS correctness,
    CAST(JSON_VALUE(outputs, "$.prediction") AS BOOLEAN) AS predicted_violating,
    JSON_VALUE(outputs, "$.confidence") AS confidence,
    CONCAT(
      IFNULL(JSON_VALUE(outputs, "$.conclusion"), ''),
      ' ',
      ARRAY_TO_STRING(JSON_EXTRACT_STRING_ARRAY(outputs, "$.reasons"), '. ', '')
    ) AS full_reasons,

    -- Record and Ground Truth
    records AS record,
    CAST(JSON_VALUE(records, '$.ground_truth.violating') AS BOOLEAN) AS expected_violating,

    -- Other
    error,
    tracing_link
FROM
    source,
    -- This is a lateral join to unnest the records array if it exists
    UNNEST(IF(JSON_QUERY_ARRAY(inputs, '$.records') IS NULL, [CAST(NULL AS JSON)], JSON_QUERY_ARRAY(inputs, '$.records'))) AS records
