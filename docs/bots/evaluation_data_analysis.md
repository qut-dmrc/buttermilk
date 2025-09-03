# Analyzing Evaluation Data

This document outlines the key data structures and workflows for analyzing A/B testing and agent evaluation results.

## Data Source & Analysis Workflow

- **Primary Data Table:** Raw agent execution data (`AgentTrace` objects) is stored in the BigQuery table: `prosocial-443205.testing.flow`.
- **Preferred Analysis Method:** The primary method for analysis is to create SQL Views in BigQuery that pre-process and aggregate the data. Downstream tools like Looker Studio should connect to these views, not the raw table. This centralizes business logic and improves performance.

## Core Evaluation Schema

To evaluate a classification agent's performance (e.g., a `JUDGE` agent), you need to find the corresponding `SCORER` agent's trace, which contains both the prediction and the ground truth.

- **Predicted Label:** The `JUDGE` agent's classification is a boolean field named `prediction` in its `outputs`. In the SQL views, this is typically accessed via `JSON_VALUE(outputs, "$.prediction")`.
- **True Label (Ground Truth):** The correct "golden set" answer is stored within the `SCORER` agent's trace. The path to the label is inside the original data record: `JSON_VALUE(records, '$.ground_truth.violating')`.

## Existing SQL Views for Analysis

The project contains pre-built SQL views to simplify building dashboards. These should be the starting point for any new analysis.

- **Location:** `buttermilk/schemas/views/`
- **`judge_scores.sql`:** A general-purpose view that joins `JUDGE` predictions with their corresponding `SCORER` evaluations. It provides high-level correctness scores and textual feedback.
- **`confusion_matrix.sql`:** A specialized view for classification analysis. It calculates a full confusion matrix (TP, TN, FP, FN) and derived metrics like Precision, Recall, and F1-Score. **Use this as the foundation for evaluating classification performance.**

## Dashboarding

- **Tool:** Looker Studio is the current tool for creating dynamic dashboards from the BigQuery views.
- **Workflow:** Connect Looker Studio directly to a BigQuery view. The view should handle all complex calculations (like F1-score), leaving Looker to do simple aggregation (SUM, AVG) and visualization.
