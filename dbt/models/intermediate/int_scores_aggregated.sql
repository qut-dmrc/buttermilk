SELECT
    -- IDs
    session_id,
    call_id AS scorer_call_id,
    parent_call_id,
    timestamp,

    -- Scorer Info
    agent_name AS scorer,
    agent_model AS scorer_model,
    template_name AS scorer_template,
    template_hash AS scorer_hash,
    agent_role AS role,

    -- Scoring Info
    correctness,
    assessments,
    tracing_link AS scorer_tracing_link
FROM
    {{ ref('stg_flows') }}
WHERE
    agent_role IN ('SCORERS')
