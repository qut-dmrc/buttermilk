from promptflow.core import tool

from datatools.utils import scrub_serializable

COL_PREDICTION='prediction'
@tool
def eval_answer(groundtruth: dict, prediction, reasons: dict = {}, scores: dict = {}, labels: list[str] = []) -> dict:
    """
    This tool processes the prediction of a single line and returns the processed result.

    :param groundtruth: the groundtruth of a single line.
    :param prediction: the prediction of a single line.
    """

    expected = groundtruth['answer']
    overall_answer = (prediction == expected)

    scored_result = dict(predicted=prediction,
                         expected=expected,
                         correct=overall_answer)
    try:
        scored_scores = {}
        pred_measures = {score['measure']: int(score['score']) for score in scores}
        expected_measures = {score['measure']: int(score['score']) for score in groundtruth.get('scores', [])}
        for measure, expected_score in expected_measures.items():
            pred_score = pred_measures.get(measure, None)
            if pred_score is not None and pred_score == expected_score:
                scored_scores[measure] = True
            else:
                scored_scores[measure] = False
        if scored_scores:
            scored_result['scores'] = scored_scores
    except (TypeError,ValueError) as e:
        # detailed scores not available
        pass

    try:
        scored_reasons = {}
        pred_criteria = {reason['heading']: int(reason['violates']) for reason in reasons}
        expected_criteria = {k: int(v) for k, v in groundtruth.get('criteria', {}).items()}
        for measure, expected_score in expected_criteria.items():
            pred_score = pred_criteria.get(measure, None)
            if pred_score is not None and pred_score == expected_score:
                scored_reasons[measure] = True
            else:
                scored_reasons[measure] = False
        if scored_reasons:
            scored_result['criteria'] = scored_reasons
    except (TypeError,ValueError) as e:
        # detailed scores not available
        pass

    try:
        labels_expected = set([str.lower(l) for l in groundtruth.get('labels', [])])
        if labels_expected:
            labels_pred = set([str.lower(l) for l in labels])
            correct_labels = list(labels_pred.intersection(labels_expected))
            missing_labels = list(labels_expected - labels_pred)
            wrong_labels = list(labels_pred - labels_expected)

            scored_result.update(dict(correct_labels=correct_labels,
                    missing_labels=missing_labels,
                    extra_labels=wrong_labels))

    except (TypeError,ValueError):
        # labels not available
        pass

    return scrub_serializable(scored_result)
