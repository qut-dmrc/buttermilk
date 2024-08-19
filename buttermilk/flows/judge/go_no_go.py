
from promptflow.core import tool


@tool
def go_no_go(text: str = '') -> bool:
    if (not isinstance(text, str)) or (not text) or (text.strip() == ""):
        return False
    return True

