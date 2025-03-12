import regex as re
import textwrap
from typing import Mapping, Sequence
from pydantic import BaseModel

SLACK_MAX_MESSAGE_LENGTH=3000


def format_response(inputs) -> list[str]:
    if not inputs:
        return []
    
    output_lines = []

    if isinstance(inputs, BaseModel):
        inputs = inputs.model_dump()

    if isinstance(inputs, str):
        output_lines += [inputs]

    elif isinstance(inputs, Mapping):
        for k, v in inputs.items():
            output_lines += [f"*{k}*: {v}\n"]

    elif isinstance(inputs, Sequence):
        output_lines.extend(inputs)

    else:
        output_lines += [f"*Unexpected response type*: {type(inputs)}"]

    output_lines = strip_and_wrap(output_lines)

    if isinstance(output_lines, str):
        output_lines = [output_lines]

    return output_lines


def strip_and_wrap(lines: list[str]) -> list[str]:
    stripped = ''

    for line in lines:
        text = re.sub(r"\n{2,+}", "\n", line)
        text = str(text) + " "  # add trailing space here, and we'll remove it in the next step if it's duplicated
        text = re.sub(r"\s{2,+}", " ", text)
        stripped += text

    # Break output into a list of strings of max length 3000 characters, wrapping nicely
    return textwrap.wrap(stripped, width=SLACK_MAX_MESSAGE_LENGTH, replace_whitespace=False)  # Preserve newlines