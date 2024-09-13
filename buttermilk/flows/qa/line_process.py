from promptflow.core import tool

from buttermilk.flows.common.config import COL_PREDICTION


@tool
def eval_answer(groundtruth: dict, result: dict) -> dict:
    """
    This tool processes the prediction of a single line and returns the processed result.

    :param groundtruth: the groundtruth of a single line.
    :param response: the prediction of a single line.
    """
    expected = groundtruth['answer']
    predicted = result[COL_PREDICTION]
    overall_answer = (predicted == expected)

    scored_result = dict(predicted=predicted,
                         expected=expected,
                         correct=overall_answer)

    return scored_result