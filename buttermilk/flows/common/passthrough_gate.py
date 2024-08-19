
from promptflow.core import tool


@tool
def passtrough(text: str) -> str:
    return text

