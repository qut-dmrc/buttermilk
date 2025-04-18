from json import JSONDecodeError
from typing import Any, Literal

import json_repair
import regex as re
from pydantic import BaseModel, Field

from buttermilk.utils.utils import load_json_flexi

from .._core.log import logger


class ChatParser(BaseModel):
    """A safe JSON parser. If all else fails, return the original string as a dictionary with the key 'response'"""

    # Error handling options:
    # - raise: raise an error if the JSON is invalid
    # - warn: log a warning if the JSON is invalid
    # - ignore: ignore the error and return the original string in a dictionary with the key 'response' (default)

    on_error: Literal["raise", "warn", "ignore"] = Field(
        default="warn",
        description="Error handling options: raise, warn, or ignore",
    )

    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call to a JSON object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed JSON object.

        """
        output = dict()
        try:
            # First we're going to try to remove any text around a possible JSON string.
            # This pattern removes any whitespace after the first "{" and before the last "}".
            pat = r"\{\s*(.*)\s*\}"
            match = re.search(pat, text, re.DOTALL)
            if not match:
                raise JSONDecodeError(
                    "Unable to find JSON brackets in response",
                    doc=text,
                    pos=0,
                )
            json_str = "{" + match.group(1) + "}"
            try:
                output = load_json_flexi(json_str)
            except (JSONDecodeError, ValueError):
                output = json_repair.loads(json_str)

        except JSONDecodeError as e:
            if self.on_error == "raise":
                logger.error(f"Unable to decode JSON in result: {text}")
                raise e
            logger.warning(f"Unable to decode JSON in result: {text}")
            return dict(response=text, error="Unable to decode JSON in result")

        if not isinstance(output, dict):
            logger.warning(f"Unable to decode JSON in result: {text}")
            output = dict(response=output)

        output = convert_dict_types(output)

        return output


def convert_dict_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_dict_types(v)
        return obj
    if isinstance(obj, list):
        return [convert_dict_types(item) for item in obj]
    if isinstance(obj, str):
        if str.lower(obj) == "true":
            return True
        if str.lower(obj) == "false":
            return False
        if obj.isdigit():
            return int(obj)
        try:
            float_v = float(obj)
            if int(float_v) == float_v:
                return int(float_v)
            return float_v
        except ValueError:
            pass

    return obj
