from typing import List
import numpy as np
from promptflow.core import log_metric, tool
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, log_loss,precision_recall_fscore_support
from buttermilk.utils.utils import scrub_serializable

@tool
def aggregate(processed_results: List[dict]):
    """
    This tool aggregates the processed result of all lines and calculate the accuracy. Then log metric for the accuracy.

    :param processed_results: List of the output of line_process node.
    """

    # Discard empty records
    results = [ r for r in processed_results if r and 'correct' in r.keys() ]

    n = len(results)

    metrics = dict(n=n)
    log_metric(key="n", value=n)

    if n <= 1:
        # nothing useful to be done here
        return metrics

    # Calculate classification results from scikit-learn stats
    pred = np.array([x.get('predicted') for x in results])
    expected = np.array([x.get('expected') for x in results])

    try:
        alignment = np.mean([x.get('alignment') for x in results if x.get('alignment')])
        metrics['mean_alignment']=alignment
    except ValueError:
        alignment = None

    #accuracy = round((sum([1 for x in results if x.get('correct')]) / len(results)), 2)
    try:
        accuracy = accuracy_score(expected, pred)
        if not np.isnan(accuracy):
            metrics['accuracy'] = accuracy
    except ValueError:
        accuracy = None

    try:
        for key, value in zip(["tn", "fp", "fn", "tp"], confusion_matrix(expected, pred).ravel()):
            if np.isnan(value):
                value = 0
            metrics[key] = value
    except ValueError:
        pass

    try:
        roc_auc = roc_auc_score(expected, pred)

        if not np.isnan(roc_auc):
            metrics['roc_auc_score'] = roc_auc
    except ValueError:
        roc_auc = None

    # Calculate the log loss
    try:
        log_loss_value = log_loss(expected, pred)
        if not np.isnan(log_loss_value):
            metrics['log_loss_value'] = log_loss_value
    except ValueError:
        log_loss_value = None

    try:
        precision, recall, f1, support = precision_recall_fscore_support(expected, pred, average='weighted')
        if not np.isnan(precision):
            metrics['precision'] = precision
        if not np.isnan(recall):
            metrics['recall'] = recall
        if not np.isnan(f1):
            metrics['f1'] = f1
    except ValueError:
        pass

    metrics_converted = scrub_serializable(metrics)
    metrics_converted = { k:v for k,v in metrics_converted.items() if v is not None}
    for k, v in metrics_converted.items():
        log_metric(key=k, value=v)
    return metrics_converted
