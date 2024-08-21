from promptflow.core import tool

COL_PREDICTION='prediction'
@tool
def validate_inputs(groundtruth: dict, prediction: dict) -> bool:
    try:
        if not prediction or prediction.get('error') or COL_PREDICTION not in prediction:
            return False
        if not groundtruth or 'answer' not in groundtruth:
            return False
    except Exception as e:
        return False
    return True