"""Utility functions for extracting data from various message types in Buttermilk.

This module provides helper functions, primarily `extract_message_data`, which
is used by Buttermilk agents (e.g., in their `_listen` method) to parse incoming
messages and extract relevant information based on predefined JMESPath mappings.
This allows for flexible and configurable data extraction from diverse message
structures without hardcoding access patterns within the agent logic.
"""

from collections.abc import Sequence # For type hinting sequences
from typing import Any # For general type hinting

import jmespath  # For resolving input mappings using JMESPath query language
from jmespath import exceptions as jmespath_exceptions # JMESPath specific exceptions

from buttermilk._core.contract import GroupchatMessageTypes # Union type for messages
from buttermilk._core.log import logger # Centralized logger
from buttermilk.utils.utils import clean_empty_values # Utility to remove empty values from dicts


def extract_message_data(
    message: GroupchatMessageTypes,
    source: str,
    input_mappings: dict[str, str],
) -> dict[str, Any]:
    """Extracts data from an incoming message object using JMESPath expressions.

    This function takes a message (typically a Pydantic model from
    `buttermilk._core.contract`), a source identifier, and a dictionary of
    input mappings. Each mapping consists of a target key and a JMESPath
    expression. The function serializes the message to a dictionary, makes it
    accessible under a key derived from `source`, and then applies each JMESPath
    expression to extract data.

    The results are cleaned to remove None or empty values (empty lists/dicts)
    before being returned.

    Args:
        message (GroupchatMessageTypes): The incoming message object to extract
            data from. This is typically a Pydantic model instance.
        source (str): An identifier for the message sender or source. The first
            part of this string (before any hyphen) is used as the top-level
            key in the data dictionary against which JMESPath queries are run.
        input_mappings (dict[str, str]): A dictionary where keys are the desired
            names for the extracted data, and values are JMESPath expressions
            used to query the message data.

    Returns:
        dict[str, Any]: A dictionary containing the extracted key-value pairs.
        Keys are from `input_mappings`, and values are the results of the
        corresponding JMESPath queries after cleaning. If a query yields no
        result or an empty result, the key might be absent from the output dict.
        Returns an empty dictionary if `input_mappings` is empty or invalid.

    Example:
        If `message` is an `AgentOutput` from agent "Summarizer-123" with
        `message.outputs = {"summary_text": "This is a summary."}`, and
        `input_mappings` is `{"extracted_summary": "Summarizer.outputs.summary_text"}`,
        the function would return `{"extracted_summary": "This is a summary."}`.
    """
    extracted_data: dict[str, Any] = {}
    if not input_mappings or not isinstance(input_mappings, dict):
        logger.debug("No valid input mappings provided to extract_message_data; returning empty dict.")
        return extracted_data

    # Prepare the data dictionary for JMESPath search.
    # The message content is nested under a key derived from the source.
    # e.g., if source is "AgentName-xyz", key becomes "AgentName".
    source_key = source.split("-", maxsplit=1)[0]
    try:
        # model_dump() is preferred for Pydantic v2
        message_dict = message.model_dump(mode="json") # Serialize to dict, handling complex types
    except AttributeError: # Fallback for older Pydantic or non-Pydantic objects if any
        try:
            message_dict = message.dict() # type: ignore
        except AttributeError:
            logger.error(f"Message object of type {type(message)} does not have model_dump or dict method.")
            return extracted_data # Cannot process further

    data_for_jmespath = {source_key: message_dict}

    # Iterate through the input mappings and apply JMESPath expressions
    for target_key, jmespath_expr in input_mappings.items():
        if jmespath_expr and isinstance(jmespath_expr, str):
            try:
                search_result = jmespath.search(jmespath_expr, data_for_jmespath)
                
                # Clean up search_result: remove None or empty sequences/mappings from list results
                if isinstance(search_result, Sequence) and not isinstance(search_result, str):
                    cleaned_sequence = [item for item in search_result if item is not None and item != [] and item != {}]
                    if not cleaned_sequence: # If list becomes empty after cleaning
                        search_result = None # Treat as no result
                    else:
                        search_result = cleaned_sequence
                
                # Store if JMESPath found something meaningful (not None, not an empty list/dict after cleaning)
                if search_result is not None and search_result != [] and search_result != {}:
                    extracted_data[target_key] = search_result
                else:
                    logger.debug(f"JMESPath expression '{jmespath_expr}' for key '{target_key}' yielded no meaningful result.")

            except jmespath_exceptions.JMESPathError as e: # Catch specific JMESPath errors
                logger.warning(
                    f"Error applying JMESPath expression '{jmespath_expr}' for key '{target_key}': {e!s}. Skipping this mapping."
                )
            except Exception as e: # Catch any other unexpected errors during search
                logger.error(
                    f"Unexpected error during JMESPath search for key '{target_key}' with expression '{jmespath_expr}': {e!s}. Skipping."
                )
        else:
            logger.warning(
                f"Invalid or empty JMESPath expression for key '{target_key}': '{jmespath_expr}'. Skipping this mapping."
            )
    
    # Final cleaning of the entire extracted dictionary (e.g., if some values were initially non-empty then became so)
    # The clean_empty_values utility might be redundant if individual search_results are already well-filtered.
    # However, keeping it provides an additional layer of cleanup.
    final_cleaned_data = clean_empty_values(extracted_data)

    if final_cleaned_data:
        logger.debug(f"Finished extracting data. Keys extracted: {list(final_cleaned_data.keys())}")
    else:
        logger.debug("Finished extracting data. No data was extracted based on the provided mappings.")
        
    return final_cleaned_data
