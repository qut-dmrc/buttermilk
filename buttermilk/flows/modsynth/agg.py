
from typing import List
from promptflow import tool

import uuid

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def aggregate(model1: dict, model2: dict, model3: dict):
    """
    This tool aggregates the processed result of all lines and log metric.
    :param processed_results: List of the output of line_process node.
    """
    # Add your aggregation logic here
    aggregated_results = []
    for result in [model1, model2, model3]:
        reasons = [f'{k}: {str(result.get(k))}' for k in ['interpretation', 'reasons', 'conclusion', 'predicted']]
        text = '\n'.join(reasons)
        answer = dict(text=text, id=uuid.uuid4().__str__())
        aggregated_results.append(answer)

    return dict(answers=aggregated_results)
