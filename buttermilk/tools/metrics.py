from re import T
import  pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from promptflow.core import log_metric
class Scorer:
    def __call__(self, df: pd.DataFrame, *, col: str, groundtruth: str = 'expected', prediction: str = 'prediction'):

        df.loc[:, 'preds'] = pd.json_normalize(df[col])[prediction].astype(int).to_numpy()
        df['correct'] = df.apply(lambda x: x[col][prediction] == x[groundtruth]['answer'], axis='columns')
        return df


class Metriciser:
    def evaluate_results(self, dataset, col: str, levels: list[str] = [], groundtruth: str = 'expected', prediction: str = 'prediction', unique_col:str='timestamp') -> pd.DataFrame:
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
        df = dataset.copy()
        df_results = pd.json_normalize(df[col])
        df_results.loc[:, 'preds'] = df_results[prediction].astype(int).to_numpy()
        df_results.loc[:, 'expected'] = pd.json_normalize(df[groundtruth])['answer'].astype(int).to_numpy()
        df_results.loc[:, 'correct'] = (df_results['preds'] == df_results['expected'])
        idx_cols = df_results.index.names
        if unique_col not in idx_cols:
            df_results = df_results.set_index(unique_col, append=True)
            df_results = df_results.loc[~(df_results.index.duplicated()),:]
        df_results = df_results.reset_index(unique_col, drop=False)

        df_results = df_results.set_index(levels)
        if levels:
            grouper = df_results.groupby(level=levels)
        else:
            grouper = df_results.groupby(level=df_results.index.names)

        acc = grouper.agg(n=(unique_col, "nunique"))

        # Compute the metrics
        acc["precision"] = grouper.apply(
            lambda x: precision_score(df_results['expected'], df_results['preds'], zero_division=0)
        )
        acc["recall"] = df.groupby(level=levels).apply(
            lambda x: recall_score(df_results['expected'], df_results['preds'], zero_division=0)
        )
        acc["f1-score"] = df.groupby(level=levels).apply(
            lambda x: f1_score(df_results['expected'], df_results['preds'], zero_division=0)
        )
        acc["accuracy"] = df.groupby(level=levels).apply(
            lambda x: accuracy_score(df_results['expected'], df_results['preds'])
        )

        # Calculate the confusion matrix for each level of the multi-index
        cm = (
            grouper.apply(lambda x: pd.crosstab(df_results['expected'], df_results['preds']))
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
        try:  # try to log metrics to promptflow
            for row in acc.to_dict(orient='records'):
                for key, value in row.items():
                    log_metric(key=key, value=value)
        except:
            pass

        return acc
