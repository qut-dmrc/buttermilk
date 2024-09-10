from promptflow.core import tool


@tool
def eval_answer(groundtruth: dict, predicted: int) -> dict:
    """
    This tool processes the prediction of a single line and returns the processed result.

    :param groundtruth: the groundtruth of a single line.
    :param predicted: the prediction of a single line.
    :param expected_reasons: a list of key points of correct reasoning
    """
    expected = groundtruth['answer']
    overall_answer = (predicted == expected)

    scored_result = dict(predicted=predicted,
                         expected=expected,
                         correct=overall_answer,
                         groundtruth=expected)

    return scored_result
