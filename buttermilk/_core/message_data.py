"""Utility functions for extracting data from messages in Buttermilk agents.
"""

from collections.abc import Sequence
from typing import Any

import jmespath  # For resolving input mappings
from jmespath import exceptions as jmespath_exceptions
from pydantic import BaseModel

from buttermilk._core.contract import GroupchatMessageTypes
from buttermilk._core.log import logger
from buttermilk._core.types import Record


def extract_message_data(
    message: GroupchatMessageTypes,
    source: str,
    input_mappings: dict[str, str],
) -> dict[str, Any]:
    """Extracts data from a message based on provided input mappings.
    
    Args:
        message: The incoming message object
        source: Identifier of the message sender
        input_mappings: Dictionary of key -> JMESPath expression mappings
        
    Returns:
        A dictionary containing the extracted key-value pairs

    """
    extracted: dict[str, Any] = {}
    if not input_mappings or not isinstance(input_mappings, dict):
        logger.debug("No valid input mappings provided")
        return extracted

    # Create data dictionary with source as key
    datadict = {source.split("-", maxsplit=1)[0]: message.model_dump()}

    # Iterate through the input mappings
    for key, mapping in input_mappings.items():
        if mapping and isinstance(mapping, str):
            try:
                # Use JMESPath to search the datadict based on the mapping expression
                search_result = jmespath.search(mapping, datadict)
                if search_result and isinstance(search_result, Sequence) and not isinstance(search_result, str):
                    # Remove None or empty results
                    search_result = [x for x in search_result if x is not None and x != [] and x != {}]

                if search_result is not None and search_result != [] and search_result != {}:
                    # Store if JMESPath found something (could be False, 0, etc.)
                    extracted[key] = search_result
                else:
                    logger.debug(f"Mapping '{mapping}' for key '{key}' yielded None.")

            except jmespath_exceptions.ParseError:
                # If the mapping is just a plain string, not a JMESPath expression,
                # Skip non-JMESPath strings
                continue
            except Exception as e:
                logger.warning(f"Error applying JMESPath mapping '{mapping}' for key '{key}': {e}")
        else:
            # Handle non-string or empty mappings
            logger.error(f"Invalid or complex input mapping for key '{key}': {mapping}. Skipping.")

    logger.debug(f"Finished extracting vars. Keys extracted: {list(extracted.keys())}")
    return extracted


def extract_records_from_message(data: Record | dict[str, Any]) -> list[Record]:
    """Extract Record objects from the 'records' field in the extracted data.
    
    Args:
        data: Record or dictionary containing 'records' key
        
    Returns:
        List of Record objects

    """
    if isinstance(data, Record):
        return [data]
    if hasattr(data, "model_dump"):
        data = data.model_dump()

    record_data = data.get("records", [])

    records = []
    # Handle both single record and list of records
    if not isinstance(record_data, Sequence) or isinstance(record_data, str):
        record_data = [record_data]

    for rec in record_data:
        try:
            if isinstance(rec, Record):
                records.append(rec)
            elif isinstance(rec, dict):
                records.append(Record(**rec))
            elif isinstance(rec, BaseModel):
                records.append(Record(**rec.model_dump()))
        except Exception as e:
            logger.warning(f"Error creating Record object from data: {e}")

    return records
