from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

def evaluate_results(df, levels: list[str]) -> pd.DataFrame:
    """
    Evaluates the results of one or more training runs,
    aggregating at the levels selected.

    Args:
        df (pd.DataFrame): The dataframe with the results. This should be fully indexed with
                            unique observations in long form. It should contain at least the
                            fields 'expected_' and 'result', as well as a unique 'timestamp'.

        levels (list[str]): The levels of the dataframe to use for aggregation.

    Returns:
        pd.DataFrame: The updated DataFrame with the new metric columns.
    """
    df = df.copy()[["expected", "predicted"]].astype(int)

    if "timestamp" in df.index.names:
        df = df.reset_index(level=["timestamp"])

    idx_cols = df.index.names

    grouper = df.groupby(level=levels)

    acc = grouper.agg(n=("timestamp", "nunique"))
    # Compute the metrics
    acc["precision"] = grouper.apply(
        lambda x: precision_score(x["expected"], x["predicted"], zero_division=0)
    )
    acc["recall"] = df.groupby(level=levels).apply(
        lambda x: recall_score(x["expected"], x["predicted"], zero_division=0)
    )
    acc["f1-score"] = df.groupby(level=levels).apply(
        lambda x: f1_score(x["expected"], x["predicted"], zero_division=0)
    )
    acc["accuracy"] = df.groupby(level=levels).apply(
        lambda x: accuracy_score(x["expected"], x["predicted"])
    )

    # Calculate the confusion matrix for each level of the multi-index
    cm = (
        grouper.apply(lambda x: pd.crosstab(x["expected"], x["predicted"]))
        .unstack(level="expected", fill_value=0, sort=True)
        .sort_index(axis="columns")
    )

    cm.columns = [f"predicted_{n[0]}_expected_{n[1]}".lower() for n in cm.columns]
    rename_dict = {
        "predicted_0_expected_0": "TN",
        "predicted_0_expected_1": "FN",
        "predicted_1_expected_0": "FP",
        "predicted_1_expected_1": "TP",
    }
    cm = cm.rename(columns=rename_dict)
    proportion = cm.apply(lambda x: x / x.sum(), axis=1)
    proportion.columns = [f"{n}_proportion" for n in proportion.columns]
    acc = pd.concat([acc, cm, proportion], axis=1)

    return acc
