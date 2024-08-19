
from typing import Optional

from promptflow.core import tool

from datatools.utils import scrub_serializable


@tool
def prepare_result(source: str, result: dict, record_id: Optional[str] = None) -> dict:
    result['record_id'] = record_id
    result['source'] = source
    return scrub_serializable(result)
