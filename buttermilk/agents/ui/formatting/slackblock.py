import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import regex as re
from pydantic import BaseModel

from buttermilk._core.contract import AgentOutput
from buttermilk._core.defaults import SLACK_MAX_MESSAGE_LENGTH
from buttermilk.agents.evaluators.scorer import QualScore


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
            if v:
                output_lines += [f"*{k}*: {v}\n"]

    elif isinstance(inputs, Sequence):
        output_lines.extend([x for x in inputs if x])

    else:
        output_lines += [f"*Unexpected response type*: {type(inputs)}"]

    output_lines = strip_and_wrap(output_lines)

    if isinstance(output_lines, str):
        output_lines = [output_lines]

    return [x for x in output_lines if x]


def strip_and_wrap(lines: list[str]) -> list[str]:
    stripped = ""

    for line in lines:
        text = re.sub(r"\n{2,+}", "\n", line)
        text = (
            str(text) + " "
        )  # add trailing space here, and we'll remove it in the next step if it's duplicated
        text = re.sub(r"\s{2,+}", " ", text)
        stripped += text

    # Break output into a list of strings of max length 3000 characters, wrapping nicely
    return textwrap.wrap(
        stripped,
        width=SLACK_MAX_MESSAGE_LENGTH,
        replace_whitespace=False,
    )  # Preserve newlines

# Add this helper function after the strip_and_wrap function


def create_context_blocks(elements_list, max_elements_per_block=10):
    """Create Slack context blocks with element chunking to respect Slack's limits.

    Args:
        elements_list: List of mrkdwn text elements to include
        max_elements_per_block: Maximum elements per context block (Slack limit is 10)

    Returns:
        List of context block dictionaries ready for Slack API

    """
    blocks = []

    # Process elements in chunks respecting Slack's element limit
    for i in range(0, len(elements_list), max_elements_per_block):
        chunk = elements_list[i : i + max_elements_per_block]
        blocks.append({
            "type": "context",
            "elements": chunk,
        })

    return blocks


def blocks_with_icon(
    data: dict,
    keys_to_icon_map: list[tuple[str, str | tuple[str, str]]],
) -> dict[str, Any]:
    """Convert matching keys to formatted blocks with icons"""
    for key, icon in keys_to_icon_map:
        special_text = []
        if key in data:
            value = data.pop(key)
            if value is not None:
                # check for binary icons
                if isinstance(icon, tuple):
                    icon = icon[0] if value else icon[1]
                special_text.append(f"{icon} *{key.capitalize()}:* {value}")
        if special_text:
            return {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(special_text),
                },
            }
    return {}


def dict_to_blocks(input) -> list[dict]:
    """Convert a dict to a list of blocks with mrkdwn elements"""
    elements = [{"type": "mrkdwn", "text": text} for text in format_response(input)]
    return create_context_blocks(elements_list=elements)


def format_slack_message(result: AgentOutput) -> dict:
    """Format message for Slack API with attractive blocks for structured data"""

    blocks = []

    result_copy = result.model_copy(deep=True)

    # fix this later -- handling for dict and result objects:
    if not isinstance(result_copy.outputs, dict):
        result_copy.outputs = result_copy.outputs.model_dump()

    # Add header with model identifier
    header_text = (
        f":robot_face: {result_copy.parameters.get('name', '')} {result_copy.parameters.get('agent_id', '')}  {result_copy.parameters.get('model', '')} ".strip()
    )
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": header_text,
            "emoji": True,
        },
    })

    # Handle error case
    if result.error:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Error!*\n" + "\n".join(format_response(result.error)),
            },
        })

    else:
        if isinstance(result_copy.outputs, QualScore):
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": str(result_copy.outputs),
                    },
                }
            )
        else:
            icontext = blocks_with_icon(
                result_copy.parameters,
                [
                    ("input", ":mag:"),
                    ("criteria", ":clipboard:"),
                ],
            )
            if icontext:
                blocks.append(icontext)

            judge_results = dict(
                prediction=result_copy.outputs.pop("prediction", None),
                confidence=result_copy.outputs.pop("confidence", None),
                severity=result_copy.outputs.pop("severity", None),
            )
            icontext = blocks_with_icon(
                judge_results,
                [
                    ("prediction", (":biohazard_sign:", ":ring_buoy:")),
                    ("confidence", ":bar_chart:"),
                    ("severity", ":warning:"),
                ],
            )
            if icontext:
                blocks.append(icontext)

            # Add labels as chips if present

            if labels := result_copy.outputs.pop("labels", None):
                label_text = "*Labels:* " + " ".join([f"`{label}`" for label in labels if label])
                if label_text:
                    blocks.append(
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": label_text,
                            },
                        }
                    )

            # Extract reasons for special handling
            reasons = result_copy.outputs.pop("reasons", []) if isinstance(result_copy.outputs, dict) else []

            # Handle any remaining fields in outputs
            blocks.extend(dict_to_blocks(result_copy.outputs))

            # Add divider before reasons if there are any
            if reasons:
                blocks.append(
                    {
                        "type": "divider",
                    }
                )

                # Add each reason as its own contextual block for better readability

                # Convert reasons to mrkdwn elements
                reason_elements = [{"type": "mrkdwn", "text": f"{i + 1}. {reason}"} for i, reason in enumerate(reasons)]
                # Add chunked context blocks
                blocks.extend(create_context_blocks(reason_elements))

            # Handle records if present
            if hasattr(result_copy, "records") and result_copy.records:
                # Process each record
                for record in result_copy.records:
                    record_metadata = dict(record.metadata)
                    record_metadata["record_id"] = record.record_id
                    record_metadata["title"] = record_metadata.get("title") or record.title

                    # Extract metadata
                    icontext = blocks_with_icon(
                        record.metadata,
                        keys_to_icon_map=[
                            ("title", ":bookmark:"),
                            ("outlet", ":newspaper:"),
                            ("date", ":calendar:"),
                            ("url", ":link:"),
                            ("record_id", ":id:"),
                        ],
                    )

                    if icontext:
                        blocks.append(icontext)

                    elements = []

                    # Split text into chunks of ~3000 chars
                    chunk_size = 2950
                    for i in range(0, len(record.content), chunk_size):
                        chunk = record.content[i : i + chunk_size]
                        elements.append(
                            {
                                "type": "mrkdwn",
                                "text": chunk,
                            }
                        )
                    blocks.extend(create_context_blocks(elements))

    # Slack has a limit on blocks, so ensure we don't exceed it
    blocks = blocks[:50]  # Slack's block limit

    # If we didn't extract content earlier, do it now:
    if len(blocks) == 1 and result_copy.content:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": result_copy.content,
                },
            }
        )
    return {
        "blocks": blocks,
        "text": result_copy.content,
    }


def confirm_options(
    options: list[str],
    message="Select an option:",
    placeholder="Choose an option...",
    action_id="option_selection",
    extra_blocks: list[dict] | None = None,
) -> dict:
    """Format a selection block for Slack with dropdown menu for multiple options.

    Args:
        message: The selection prompt to display
        options: List of string options for the user to choose from
        placeholder: Text to show in the dropdown before selection
        action_id: ID for the action, useful for handling responses

    Returns:
        dict: Slack blocks format for a selection menu

    """
    # Convert string options to Slack option objects
    slack_options = [
        {
            "text": {
                "type": "plain_text",
                "text": option,
                "emoji": True,
            },
            "value": f"option_{i}",  # Use index as value for tracking selected option
        }
        for i, option in enumerate(options)
    ]
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":thinking_face: *{message}*",
            },
        },
    ]
    if extra_blocks:
        blocks.extend(extra_blocks)
    blocks.extend([
        {
            "type": "actions",
            "elements": [
                {
                    "type": "static_select",
                    "placeholder": {
                        "type": "plain_text",
                        "text": placeholder,
                        "emoji": True,
                    },
                    "options": slack_options,
                    "action_id": action_id,
                },
            ],
        },
    ])
    return {
        "blocks": blocks,
        "text": f"Selection required: {message}",  # Fallback text
    }


def confirm_bool(
    message="Do you want to proceed?",
    yes_text="Yes",
    no_text="No",
    cancel=True,
    extra_blocks: list[dict] | None = None,
) -> dict:
    """Format a confirmation block for Slack with Yes/No buttons and decorative elements.

    Args:
        message: The confirmation question to display
        yes_text: Text for the confirmation button
        no_text: Text for the decline button
        extra_blocks:

    Returns:
        dict: Slack blocks format for a confirmation message

    """
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":question: *{message}*",
            },
        },
    ]
    if extra_blocks:
        blocks.extend(extra_blocks)

    blocks.extend(
        [
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": f":white_check_mark: {yes_text}",
                            "emoji": True,
                        },
                        "style": "primary",
                        "value": "confirm",
                        "action_id": "confirm_action",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": f":thinking_face: {no_text}",
                            "emoji": True,
                        },
                        # No style means default gray button
                        "value": "decline",
                        "action_id": "decline_action",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": ":x: Cancel",
                            "emoji": True,
                        },
                        "style": "danger",
                        "value": "cancel",
                        "action_id": "cancel_action",
                    },
                ],
            },
        ],
    )
    return {
        "blocks": blocks,
        "text": f"Confirmation required: {message}",  # Fallback text
    }
