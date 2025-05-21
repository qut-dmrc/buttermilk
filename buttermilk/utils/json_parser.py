from json import JSONDecodeError
from typing import Any

import json_repair
import regex as re
from pydantic import BaseModel

from buttermilk.utils.utils import load_json_flexi

from .._core.exceptions import ProcessingError
from .._core.log import logger


class ChatParser(BaseModel):
    """A JSON parser. Try to clean up the input if necessary."""

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
            raise ProcessingError from e

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
