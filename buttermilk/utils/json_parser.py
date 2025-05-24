"""Provides robust JSON parsing utilities, especially for LLM outputs.

This module includes `ChatParser` for parsing JSON that might be embedded in
text or have minor syntax errors, and `convert_dict_types` for recursively
converting stringified primitive types (bools, numbers) within nested data
structures to their actual Python types.
"""

from json import JSONDecodeError  # Standard JSON exception
from typing import Any, Literal  # For type hinting

import json_repair  # Library for repairing "broken" JSON
import regex as re  # Advanced regular expression library
from pydantic import BaseModel, Field  # Pydantic components for data validation

from buttermilk.utils.utils import load_json_flexi  # Flexible JSON loading utility

from .._core.exceptions import ProcessingError
from .._core.log import logger  # Centralized logger


class ChatParser(BaseModel):
    """A robust JSON parser designed to handle potentially messy LLM outputs.

    This parser attempts to extract and parse JSON objects that might be embedded
    within larger text strings or have minor syntax issues (e.g., trailing commas).
    It first tries to isolate a JSON-like structure using regex, then attempts
    parsing with `load_json_flexi` (which handles common issues), and falls back
    to `json_repair.loads` for more complex error correction.

    If parsing ultimately fails, behavior is determined by the `on_error` attribute.
    After successful parsing, it also attempts to convert string representations
    of booleans and numbers to their actual types using `convert_dict_types`.

    Attributes:
        on_error (Literal["raise", "warn", "ignore"]): Defines behavior upon
            JSON decoding failure:
            - "raise": Re-raises the `JSONDecodeError`.
            - "warn": Logs a warning and returns a dictionary containing the
              original text and an error message (default).
            - "ignore": Returns a dictionary containing the original text and an
              error message, without logging a warning explicitly (though underlying
              parsing attempts might log debug messages).

    """

    on_error: Literal["raise", "warn", "ignore"] = Field(
        default="warn",
        description="Defines behavior on JSON parsing failure: 'raise' an error, "
                    "'warn' and return original text, or 'ignore' and return original text.",
    )

    def parse(self, text: str) -> Any:
        r"""Parses a string, attempting to extract and decode a JSON object.

        The method employs several strategies:
        1.  Uses a regular expression `r"\{\s*(.*)\s*\}"` to find and extract content
            between the first '{' and the last '}' in the input `text`. This helps
            isolate a potential JSON object embedded in surrounding text.
        2.  Tries to parse the extracted string using `load_json_flexi`.
        3.  If `load_json_flexi` fails, it attempts parsing with `json_repair.loads`
            which can fix common JSON syntax errors.
        4.  If parsing is successful and results in a dictionary, it recursively
            converts stringified booleans and numbers to their actual types using
            `convert_dict_types`.
        5.  If parsing fails, it handles the error based on `self.on_error`.
        6.  If the parsed result is not a dictionary (e.g., a list if the JSON was an array),
            it logs a warning and wraps the result in `{"response": result}`.

        Args:
            text (str): The input string, potentially containing an embedded or
                malformed JSON object.

        Returns:
            Any: The parsed JSON object (typically a dictionary), or a dictionary
                 containing the original text and an error message if parsing fails
                 and `on_error` is not "raise".

        Raises:
            JSONDecodeError: If parsing fails and `self.on_error` is set to "raise".

        """
        parsed_output: Any = None  # Initialize with a type that can hold dict or list

        if not isinstance(text, str):
            logger.warning(f"ChatParser.parse expected a string, got {type(text)}. Attempting to stringify.")
            text = str(text)

        try:
            # Attempt to find a JSON object structure (content between first { and last })
            # This regex tries to capture content within the outermost curly braces.
            # re.DOTALL allows '.' to match newlines.
            json_block_match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)  # More robust for nested {}
            if not json_block_match:  # Fallback for array-first JSON or simpler cases
                 json_block_match = re.search(r"\[(?:[^\[\]]|(?R))*\]", text, re.DOTALL)

            if not json_block_match:  # Fallback for simple {} or [] if regex above fails
                # This simpler regex might be problematic with nested structures if not careful
                # but serves as a broader catch if the recursive ones fail or aren't matched.
                # The primary attempt is the recursive one above.
                simple_json_match = re.search(r"[\{\[]\s*(.*)\s*[\}\]]", text, re.DOTALL)
                if not simple_json_match:
                    raise JSONDecodeError("Unable to find JSON-like brackets '{...}' or '[...]' in response", doc=text, pos=0)
                json_candidate_str = simple_json_match.group(0)  # The whole match
            else:
                json_candidate_str = json_block_match.group(0)  # The whole match from recursive regex

            logger.debug(f"ChatParser: Extracted JSON candidate: {json_candidate_str[:500]}...")

            try:
                # First attempt with a flexible JSON loader
                parsed_output = load_json_flexi(json_candidate_str)
            except (JSONDecodeError, ValueError) as e1:
                logger.debug(f"ChatParser: load_json_flexi failed ({e1!s}). Trying json_repair.")
                # Fallback to json_repair for more significant errors
                parsed_output = json_repair.loads(json_candidate_str)

            logger.debug(f"ChatParser: Successfully parsed. Type: {type(parsed_output)}")

        except JSONDecodeError as e:
            raise ProcessingError from e
        
        if not isinstance(parsed_output, dict):
            logger.warning(f"Unable to decode JSON in result: {text}")
            parsed_output = dict(response=text)

        # Recursively convert stringified bools/numbers to actual types
        return convert_dict_types(parsed_output)


def convert_dict_types(obj: Any) -> Any:
    """Recursively converts string values in nested dicts/lists to Python types.

    Traverses a dictionary or list, and for each string value encountered,
    it attempts to convert it to:
    - `bool` (if string is "true" or "false", case-insensitive).
    - `int` (if string is a digit).
    - `float` (if string can be converted to float). If the float is whole
      (e.g., "5.0"), it's converted to `int`.

    Args:
        obj (Any): The dictionary, list, or other value to process.

    Returns:
        Any: The processed object with string values converted to their inferred
             Python types where possible. Non-string values or strings that cannot
             be converted are returned as is.

    """
    if isinstance(obj, dict):
        return {k: convert_dict_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_dict_types(item) for item in obj]
    if isinstance(obj, str):
        # Handle boolean strings
        obj_lower = obj.lower()
        if obj_lower == "true":
            return True
        if obj_lower == "false":
            return False
        # Handle numeric strings
        if obj.isdigit():  # Check for integer first
            return int(obj)
        try:
            # Try float, then check if it's a whole number to convert to int
            float_val = float(obj)
            if float_val.is_integer():  # Check if it's like 5.0
                return int(float_val)
            return float_val  # Keep as float
        except ValueError:
            # Not a float, so it remains a string
            pass
    return obj  # Return original if not dict, list, or convertible string
