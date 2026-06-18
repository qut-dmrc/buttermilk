-- NOTE (task buttermilk-1e23cce6): record_id + ground_truth are now read from the canonical
-- top-level `record` column (record.record_id, record.ground_truth JSON), not from
-- `inputs.$.records` (no longer emitted). Re-apply manually and VALIDATE against the live
-- table; trace schema is traces.schema.json.
WITH SCORES_AGGREGATED AS (
  -- This CTE is identical to the one in judge_scores.sql
  -- It gathers all the scores from SCORER agents.
  SELECT
    session_id,
    call_id AS scorer_call_id,
    timestamp,
    parent_call_id,
    JSON_VALUE(agent_info, "$.name") AS scorer,
    JSON_VALUE(agent_info, "$.parameters.model") AS scoring_model,
    JSON_VALUE(metadata, "$.template_hash") AS scoring_hash,
    CAST(JSON_VALUE(outputs, "$.correctness") AS FLOAT64) AS correctness
  FROM
    `prosocial-443205.testing.flow`
  WHERE
    JSON_VALUE(agent_info, "$.role") IN ('SCORERS')
),
PREDICTIONS_WITH_TRUTH AS (
  -- This CTE is modified to include the ground_truth label
  SELECT
    session_id,
    call_id,
    timestamp,
    record.record_id AS record_id,
    JSON_VALUE(agent_info, "$.name") AS judge,
    JSON_VALUE(agent_info, "$.parameters.model") AS judge_model,
    JSON_VALUE(metadata, "$.template_hash") AS judge_hash,
    JSON_VALUE(agent_info, "$.role") AS judge_role,
    -- Predicted Label
    CAST(JSON_VALUE(outputs, "$.prediction") AS BOOLEAN) AS predicted_violating,
    -- True Label (from the canonical record column; record.ground_truth is JSON)
    CAST(JSON_VALUE(record.ground_truth, '$.violating') AS BOOLEAN) AS true_violating
  FROM
    `prosocial-443205.testing.flow`
  WHERE
    JSON_VALUE(agent_info, "$.role") IN ('JUDGE', 'SYNTHESISER')
),
COMPARISON AS (
  -- This new CTE joins predictions with scores and calculates the confusion matrix cells for each prediction
  SELECT
    p.session_id,
    p.call_id,
    p.timestamp,
    p.record_id,
    p.judge,
    p.judge_model,
    p.judge_hash,
    p.judge_role,
    s.scorer,
    s.scoring_model,
    s.scoring_hash,
    s.correctness,
    p.predicted_violating,
    p.true_violating,

    -- Calculate Confusion Matrix cells for each row
    CASE WHEN p.predicted_violating = TRUE AND p.true_violating = TRUE THEN 1 ELSE 0 END AS tp,
    CASE WHEN p.predicted_violating = FALSE AND p.true_violating = FALSE THEN 1 ELSE 0 END AS tn,
    CASE WHEN p.predicted_violating = TRUE AND p.true_violating = FALSE THEN 1 ELSE 0 END AS fp,
    CASE WHEN p.predicted_violating = FALSE AND p.true_violating = TRUE THEN 1 ELSE 0 END AS fn
  FROM PREDICTIONS_WITH_TRUTH p
  LEFT JOIN SCORES_AGGREGATED s ON p.call_id = s.parent_call_id
  WHERE
    -- Data quality filters
    p.timestamp >= '2025-05-01'
    AND (s.timestamp IS NULL OR s.timestamp >= '2025-05-01')
    AND (s.timestamp IS NULL OR s.timestamp >= p.timestamp)
    AND NOT p.judge_hash IS NULL
    AND NOT s.scoring_hash IS NULL
)
-- Final aggregation to get the metrics grouped by model, template, etc.
SELECT
  judge,
  judge_model,
  SUBSTR(judge_hash, 8, 8) as judge_hash,
  judge_role,
  scorer,
  scoring_model,
  SUBSTR(scoring_hash, 8, 8) as scoring_hash,
  COUNT(*) AS total_predictions,
  SUM(tp) AS true_positives,
  SUM(tn) AS true_negatives,
  SUM(fp) AS false_positives,
  SUM(fn) AS false_negatives,
  AVG(correctness) AS avg_correctness,

  -- Key Classification Metrics
  -- Accuracy: (TP + TN) / Total
  SAFE_DIVIDE(SUM(tp) + SUM(tn), COUNT(*)) AS accuracy,

  -- Precision: TP / (TP + FP)
  SAFE_DIVIDE(SUM(tp), SUM(tp) + SUM(fp)) AS precision,

  -- Recall (Sensitivity): TP / (TP + FN)
  SAFE_DIVIDE(SUM(tp), SUM(tp) + SUM(fn)) AS recall,

  -- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
  SAFE_DIVIDE(2 * (SAFE_DIVIDE(SUM(tp), SUM(tp) + SUM(fp))) * (SAFE_DIVIDE(SUM(tp), SUM(tp) + SUM(fn))),
              (SAFE_DIVIDE(SUM(tp), SUM(tp) + SUM(fp))) + (SAFE_DIVIDE(SUM(tp), SUM(tp) + SUM(fn)))) AS f1_score,

  -- Specificity: TN / (TN + FP)
  SAFE_DIVIDE(SUM(tn), SUM(tn) + SUM(fp)) AS specificity,

  -- False Positive Rate: FP / (FP + TN)
  SAFE_DIVIDE(SUM(fp), SUM(fp) + SUM(tn)) AS false_positive_rate

FROM COMPARISON

GROUP BY
  judge,
  judge_model,
  judge_hash,
  judge_role,
  scorer,
  scoring_model,
  scoring_hash
ORDER BY
  total_predictions DESC;
