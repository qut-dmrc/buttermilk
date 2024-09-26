
from typing import List
from promptflow import tool

import uuid

@tool
def aggregate(moderated_results: dict) -> list:
    aggregated_results = []

    for name, result in moderated_results.items():
        if name in ['metadata','usage','token_usage']:
            continue
        if not isinstance(result, dict):
            continue
        if 'error' in result and result['error']:
            continue
        reasons = [f'{k}: {str(result.get(k))}' for k in ['interpretation', 'reasons', 'conclusion', 'predicted']]
        text = '\n'.join(reasons)
        answer = dict(text=text, id=f"{name}_{uuid.uuid4()}")
        aggregated_results.append(answer)

    return aggregated_results
