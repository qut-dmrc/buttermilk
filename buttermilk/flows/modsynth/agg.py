
from typing import List
from promptflow import tool

import uuid

@tool
def aggregate(**results):
    aggregated_results = []
    for name, result in results.items():
        reasons = [f'{k}: {str(result.get(k))}' for k in ['interpretation', 'reasons', 'conclusion', 'predicted']]
        text = '\n'.join(reasons)
        answer = dict(text=text, id=uuid.uuid4().__str__())
        aggregated_results.append(answer)

    return dict(answers=aggregated_results)
