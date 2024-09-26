
from typing import List
from promptflow import tool

import uuid

@tool
def aggregate(moderated_results: dict) -> list:
    breakpoint()
    aggregated_results = []

    for name, result in moderated_results.items():
        reasons = [f'{k}: {str(result.get(k))}' for k in ['interpretation', 'reasons', 'conclusion', 'predicted']]
        text = '\n'.join(reasons)
        answer = dict(text=text, id=f"{name}_{uuid.uuid4()}")
        aggregated_results.append(answer)

    return aggregated_results
