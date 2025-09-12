WITH predictions AS (
    SELECT * FROM {{ ref('int_predictions') }}
),
scores AS (
    SELECT * FROM {{ ref('int_scores_deduped') }}
)

{{
  config(
    materialized='incremental',
    unique_key='call_id',
    on_schema_change='sync_all_columns'
  )
}}

SELECT
  predictions.session_id,
  predictions.call_id,
  scores.scorer_call_id,
  predictions.timestamp,
  ARRAY_AGG(predictions.record IGNORE NULLS) AS record,
  predictions.record_id,
  predictions.judge,
  predictions.judge_model,
  predictions.judge_template,
  SUBSTR(predictions.judge_hash, 8, 8) as judge_hash,
  predictions.judge_criteria,
  predictions.judge_role,
  predictions.full_reasons,
  predictions.confidence,
  predictions.tracing_link,
  scores.scorer,
  scores.scorer_model,
  scores.scorer_template,
  SUBSTR(scores.scorer_hash, 8, 8) as scorer_hash,
  scores.role,
  scores.scorer_tracing_link,
  predictions.predicted_violating,
  predictions.expected_violating,
  CAST((predictions.predicted_violating = predictions.expected_violating) AS BOOLEAN) as correct,
  scores.correctness,
  scores.score_quality,
  ARRAY_AGG(CAST(JSON_VALUE(assessment, '$.correct') AS BOOLEAN) IGNORE NULLS) AS assessment_correct,
  ARRAY_AGG(JSON_VALUE(assessment, '$.feedback') IGNORE NULLS) AS assessment_feedback,
FROM
  predictions
LEFT JOIN
  scores ON predictions.call_id = scores.parent_call_id
LEFT JOIN
  UNNEST(scores.assessments) AS assessment
WHERE
  TRUE
  -- Filter for data after configurable cutoff date
  AND predictions.timestamp >= '{{ var("cutoff_date") }}'
  AND scores.timestamp >= '{{ var("cutoff_date") }}'
  -- This join condition can improve performance
  AND scores.timestamp >= predictions.timestamp
  
  -- Incremental logic: only process new/updated data
  {% if is_incremental() %}
    AND predictions.timestamp > (SELECT MAX(timestamp) FROM {{ this }})
  {% endif %}
GROUP BY
  predictions.session_id,
  predictions.call_id,
  scores.scorer_call_id,
  predictions.timestamp,
  predictions.record_id,
  predictions.judge,
  predictions.judge_model,
  predictions.judge_template,
  predictions.judge_hash,
  predictions.judge_criteria,
  predictions.judge_role,
  predictions.full_reasons,
  predictions.confidence,
  predictions.tracing_link,
  scores.scorer_tracing_link,
  scores.scorer,
  scores.scorer_model,
  scores.scorer_template,
  scores.scorer_hash,
  scores.role,
  scores.score_quality,
  scores.correctness,
  predictions.predicted_violating,
  predictions.expected_violating
ORDER BY
  predictions.timestamp DESC
