from json import JSONDecodeError
from typing import Any, List, Literal

import json_repair
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.utils.json import parse_json_markdown
from langchain_core.outputs import Generation
from pydantic import BaseModel, Field

import regex as re
from json import JSONDecodeError

from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown



class ChatParser(JsonOutputParser):
    """A safe JSON parser. If all else fails, return the original string as a dictionary with the key 'response'"""

    # Error handling options:
    # - raise: raise an error if the JSON is invalid
    # - warn: log a warning if the JSON is invalid
    # - ignore: ignore the error and return the original string in a dictionary with the key 'response' (default)

    on_error: Literal["raise", "warn", "ignore"] = Field(
        default="ignore", description="Error handling options: raise, warn, or ignore"
    )

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        text = "\n".join([r.text for r in result])
        output = self.parse_json(text)

        try:
            # next, we're  going to see if we have any more information in the metadata
            output["metadata"] = result[0].message.response_metadata
        except Exception as e:
            pass

        return output

    def parse_json(self, text: str):
        output = dict()
        try:
            # First we're going to try to remove any text around a possible JSON string.
            # This pattern removes any whitespace after the first "{" and before the last "}".
            pat = r"\{.*?(.*)\s*\}"
            match = re.search(pat, text, re.DOTALL)

            json_str = "{" + match.group(1) + "}"
            output = json_repair.loads(json_str)

        except Exception as e:
            try:
                output = parse_json_markdown(text)
            except (JSONDecodeError, OutputParserException, Exception) as e:
                output = dict(error="Unable to decode JSON in result", response=text)

        if not isinstance(output, dict):
            output = dict(response=output)

        output = convert_dict_types(output)

        return output



def convert_dict_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_dict_types(v)
        return obj
    elif isinstance(obj, list):
        return [convert_dict_types(item) for item in obj]
    elif isinstance(obj, str):
        if str.lower(obj) == "true":
            return True
        elif str.lower(obj) == "false":
            return False
        elif obj.isdigit():
            return int(obj)
        else:
            try:
                float_v = float(obj)
                if int(float_v) == float_v:
                    return int(float_v)
                else:
                    return float_v
            except ValueError:
                pass

    return obj