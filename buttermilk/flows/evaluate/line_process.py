from promptflow.core import tool

from flows.common import COL_PREDICTION
from datatools.utils import scrub_serializable


@tool
def eval_answer(groundtruth: dict, result: dict) -> dict:
    """
    This tool processes the prediction of a single line and returns the processed result.

    :param groundtruth: the groundtruth of a single line.
    :param prediction: the prediction of a single line.
    """
    if not result or COL_PREDICTION not in result:
        raise ValueError("Prediction not found in result")

    pred = result[COL_PREDICTION]
    expected = groundtruth['answer']
    overall_answer = (pred == expected)

    scored_result = dict(predicted=pred,
                         expected=expected,
                         correct=overall_answer)
    try:
        scores = {}
        pred_measures = {score['measure']: int(score['score']) for score in result.get('scores', [])}
        expected_measures = {score['measure']: int(score['score']) for score in groundtruth.get('scores', [])}
        for measure, expected_score in expected_measures.items():
            pred_score = pred_measures.get(measure, None)
            if pred_score is not None and pred_score == expected_score:
                scores[measure] = True
            else:
                scores[measure] = False
        if scores:
            scored_result['scores'] = scores
    except ValueError as e:
        # detailed scores not available
        pass

    try:
        reasons = {}
        pred_criteria = {reason['heading']: int(reason['violates']) for reason in result.get('reasons', [])}
        expected_criteria = {k: int(v) for k, v in groundtruth.get('criteria', {}).items()}
        for measure, expected_score in expected_criteria.items():
            pred_score = pred_criteria.get(measure, None)
            if pred_score is not None and pred_score == expected_score:
                reasons[measure] = True
            else:
                reasons[measure] = False
        if reasons:
            scored_result['criteria'] = reasons
    except ValueError as e:
        # detailed scores not available
        pass

    try:
        labels_expected = set([str.lower(l) for l in groundtruth.get('labels', [])])
        if labels_expected:
            labels_pred = set([str.lower(l) for l in result.get('labels', [])])
            correct_labels = list(labels_pred.intersection(labels_expected))
            missing_labels = list(labels_expected - labels_pred)
            wrong_labels = list(labels_pred - labels_expected)

            scored_result.update(dict(correct_labels=correct_labels,
                    missing_labels=missing_labels,
                    extra_labels=wrong_labels))

    except ValueError:
        # labels not available
        pass

    return scrub_serializable(scored_result)
