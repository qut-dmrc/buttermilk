
from promptflow.core import tool


@tool
def check_valid(content = None) -> str:
    record_text = ''
    if content and isinstance(content, str):
        record_text = content
        # TODO: handle images
    if record_text.strip() == "":
        # fail out at next step
        return ""

    return record_text.strip()
