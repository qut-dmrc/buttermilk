SELECT
    -- IDs
    session_id,
    call_id,
    record_id,
    timestamp,
    record,

    -- Judge Info
    agent_name AS judge,
    agent_model AS judge_model,
    template_name AS judge_template,
    template_hash AS judge_hash,
    judge_criteria,
    agent_role AS judge_role,

    -- Predictions
    predicted_violating,
    expected_violating,
    full_reasons,
    confidence,
    tracing_link
FROM
    {{ ref('stg_flows') }}
WHERE
    agent_role IN ('JUDGE', 'SYNTHESISER')
