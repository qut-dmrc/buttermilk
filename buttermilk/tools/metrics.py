import  pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

class Scorer:
    def __call__(self, df: pd.DataFrame, *, groundtruth: str = 'expected', prediction: str = 'prediction'):
        df['correct'] = (df[groundtruth] == df[prediction])
        return df


class Metriciser:
    def evaluate_results(self, dataset, levels: list[str] = [],  groundtruth: str = 'expected', prediction: str = 'prediction', unique_col:str='timestamp') -> pd.DataFrame:
        """
        Evaluates the results of one or more training runs,
        aggregating at the levels selected.

        Args:
            df (pd.DataFrame): The dataframe with the results. This should be fully indexed with
                                unique observations in long form.

            levels (list[str]): The levels of the dataframe to use for aggregation.
            unique_col: a column that should be unique for each observation (to avoid counting duplicates)

        Returns:
            pd.DataFrame: The updated DataFrame with the new metric columns.
        """
        df = dataset.copy()[[groundtruth, prediction]]

        idx_cols = df.index.names
        if unique_col not in idx_cols:
            df = df.set_index(unique_col, append=True)
            df = df.loc[~df.index.duplicated,:]
            df = df.reset_index(unique_col, drop=False)

        grouper = df.groupby(level=levels)

        acc = grouper.agg(n=(unique_col, "nunique"))

        # Compute the metrics
        acc["precision"] = grouper.apply(
            lambda x: precision_score(x[groundtruth], x[prediction], zero_division=0)
        )
        acc["recall"] = df.groupby(level=levels).apply(
            lambda x: recall_score(x[groundtruth], x[prediction], zero_division=0)
        )
        acc["f1-score"] = df.groupby(level=levels).apply(
            lambda x: f1_score(x[groundtruth], x[prediction], zero_division=0)
        )
        acc["accuracy"] = df.groupby(level=levels).apply(
            lambda x: accuracy_score(x[groundtruth], x[prediction])
        )

        # Calculate the confusion matrix for each level of the multi-index
        cm = (
            grouper.apply(lambda x: pd.crosstab(x[groundtruth], x[prediction]))
            .unstack(level=groundtruth, fill_value=0, sort=True)
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
