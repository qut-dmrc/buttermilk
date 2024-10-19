from re import T
import  pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from promptflow.core import log_metric

from buttermilk.flows.common.config import COL_PREDICTION
class Scorer:
    def __call__(self, df: pd.DataFrame, *, col: str, groundtruth: str = 'expected', predicted: str = COL_PREDICTION):

        df.loc[:, 'preds'] = pd.json_normalize(df[col])[predicted].astype(int).to_numpy()
        df['correct'] = df.apply(lambda x: x[col][predicted] == x[groundtruth]['answer'], axis='columns')
        return df


class Metriciser:
    def evaluate_results(self, dataset, levels: list[str] = [], groundtruth: str = 'expected', predicted: str = COL_PREDICTION, unique_col:str='job_id') -> pd.DataFrame:
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


        df_results = dataset.copy()
        idx_cols = df_results.index.names

        if unique_col in idx_cols:
            idx_cols = [x for x in idx_cols if x != unique_col]
            df_results = df_results.reset_index(unique_col, drop=False)

        if levels is not None and len(levels)>0:
            df_results = df_results.reset_index().set_index(levels)
            grouper = df_results.groupby(level=levels)
        else:
            grouper = df_results.groupby(level=idx_cols)


        # Calculate confusion matrix components
        df_results['TP'] = (df_results[groundtruth] == 1) & (df_results[predicted] == 1)
        df_results['TN'] = (df_results[groundtruth] == 0) & (df_results[predicted] == 0)
        df_results['FP'] = (df_results[groundtruth] == 0) & (df_results[predicted] == 1)
        df_results['FN'] = (df_results[groundtruth] == 1) & (df_results[predicted] == 0)

        # Aggregate confusion matrix components
        conf_matrix = grouper[['TP', 'TN', 'FP', 'FN']].sum()

        # Calculate metrics
        conf_matrix['precision'] = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP']).replace(0, 1)
        conf_matrix['recall'] = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN']).replace(0, 1)
        conf_matrix['f1-score'] = 2 * (conf_matrix['precision'] * conf_matrix['recall']) / (conf_matrix['precision'] + conf_matrix['recall']).replace(0, 1)
        conf_matrix['accuracy'] = (conf_matrix['TP'] + conf_matrix['TN']) / (conf_matrix['TP'] + conf_matrix['TN'] + conf_matrix['FP'] + conf_matrix['FN']).replace(0, 1)

        # Calculate proportions
        proportion = conf_matrix[['TP', 'TN', 'FP', 'FN']].div(conf_matrix[['TP', 'TN', 'FP', 'FN']].sum(axis=1), axis=0)
        proportion.columns = [f"{col}_proportion" for col in proportion.columns]

        # Add number of examples
        conf_matrix['n'] = grouper.agg(n=(unique_col, 'nunique'))['n']

        # Combine metrics and proportions
        metrics = pd.concat([conf_matrix, proportion], axis=1)

        metrics = metrics.sort_values(by='f1-score', ascending=False)

        return metrics


    def log_metrics(self, metrics: pd.DataFrame):
        try:  # try to log metrics to promptflow
            for row in metrics.to_dict(orient='records'):
                for key, value in row.items():
                    log_metric(key=key, value=value)
        except:
            pass
