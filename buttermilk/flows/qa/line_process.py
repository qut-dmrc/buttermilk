from promptflow.core import tool


@tool
def eval_answer(groundtruth: dict, result: dict) -> dict:
    """
    This tool processes the prediction of a single line and returns the processed result.

    :param groundtruth: the groundtruth of a single line.
    :param response: the prediction of a single line.
    """
    expected = groundtruth['answer']
    prediction = result['prediction']
    overall_answer = (prediction == expected)

    scored_result = dict(predicted=prediction,
                         expected=expected,
                         correct=overall_answer)

    return scored_result
