import textwrap
from collections.abc import Mapping, Sequence

import regex as re
from pydantic import BaseModel

from buttermilk._core.contract import AgentOutput
from buttermilk.defaults import SLACK_MAX_MESSAGE_LENGTH


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


def format_slack_message(result: AgentOutput) -> dict:
    result_copy = result.model_copy()
    """Format message for Slack API with attractive blocks for structured data"""
    blocks = []

    # Add header with model identifier
    header_text = (
        f":robot_face: {result_copy.agent_name} {result_copy.metadata.get('model')}"
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
    if any(result_copy.error):
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Error!*\n" + "\n".join(format_response(result_copy.error)),
            },
        })

    else:
        # Handle feedback metadata fields if present
        feedback_fields = ["mistake", "intervention"]
        if any(result_copy.outputs.get(k) for k in feedback_fields):
            feedback_text = ""

            # Format mistake field with special styling
            if "mistake" in result_copy.outputs:
                mistake = result_copy.outputs.get("mistake")
                icon = (
                    ":x:"
                    if mistake and mistake not in [False, "False"]
                    else ":white_check_mark:"
                )
                feedback_text += f"{icon} *Mistake:* {mistake!s}\n\n"

            # Format intervention field
            if "intervention" in result_copy.outputs and result_copy.outputs.get(
                "intervention",
            ):
                intervention = result_copy.outputs.get("intervention")
                feedback_text += f"*Intervention:*\n{intervention}\n"

            if feedback_text:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": feedback_text.strip(),
                    },
                })
        for k, v in result_copy.metadata.items():
            match k:
                case "criteria":
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f":clipboard: *Criteria:* {v}",
                        },
                    })

        if isinstance(result_copy.outputs, dict):
            # Format prediction, confidence and severity with special styling if present
            key_fields = ["prediction", "confidence", "severity"]
            if any(k in result.outputs for k in key_fields):
                special_text = ""

                if "prediction" in result.outputs:
                    prediction = result.outputs.pop("prediction", None)
                    icon = ":biohazard_sign:" if prediction else ":white_check_mark:"
                    special_text += f"{icon} *Prediction:* {prediction!s}\n"

                if "confidence" in result.outputs:
                    confidence = result.outputs.pop("confidence", "")
                    special_text += f":bar_chart: *Confidence:* {confidence}\n"

                if "severity" in result.outputs:
                    severity = result.outputs.pop("severity", "")
                    special_text += f":warning: *Severity:* {severity}\n"

                if special_text:
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": special_text.strip(),
                        },
                    })

            # Add labels as chips if present
            if labels := result.outputs.get("labels"):
                label_text = "*Labels:* " + " ".join([
                    f"`{label}`" for label in labels if label
                ])
                if label_text:
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": label_text,
                        },
                    })

            # Extract reasons for special handling
            reasons = (
                result_copy.outputs.pop("reasons", [])
                if isinstance(result_copy.outputs, dict)
                else []
            )

            # Handle any remaining fields in outputs
            if text_list := format_response(result_copy.outputs):
                # Convert text items to mrkdwn elements
                mrkdwn_elements = [
                    {"type": "mrkdwn", "text": text} for text in text_list if text
                ]
                # Add chunked context blocks
                blocks.extend(create_context_blocks(mrkdwn_elements))

            # Add divider before reasons if there are any
            if reasons:
                blocks.append({
                    "type": "divider",
                })

                # Add each reason as its own contextual block for better readability

                # Convert reasons to mrkdwn elements
                reason_elements = [
                    {"type": "mrkdwn", "text": f"{i + 1}. {reason}"}
                    for i, reason in enumerate(reasons)
                ]
                # Add chunked context blocks
                blocks.extend(create_context_blocks(reason_elements))

        else:
            # Handle case where outputs is not a dict
            for text in format_response(result.outputs):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text,
                    },
                })

        # Handle records if present
        if hasattr(result_copy, "records") and result_copy.records:
            # Process each record
            for record in result_copy.records:
                # Extract metadata
                metadata = record.metadata
                title = metadata.get("title", "Untitled")
                outlet = metadata.get("outlet", "Unknown Source")
                date = (
                    metadata.get("date", "").split("T")[0]
                    if metadata.get("date")
                    else ""
                )
                url = metadata.get("url", "")
                record_id = record.record_id

                # Format title and source info
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{title}*\n{outlet} | {date} | <{url}|View Source> | ID: `{record_id}`",
                    },
                })

                elements = []
                for para in record.paragraphs:
                    # Split text into chunks of ~3000 chars
                    chunk_size = 2950
                    for i in range(0, len(para), chunk_size):
                        chunk = para[i : i + chunk_size]
                        elements.append({
                            "type": "mrkdwn",
                            "text": chunk,
                        })

    # Slack has a limit on blocks, so ensure we don't exceed it
    blocks = blocks[:50]  # Slack's block limit

    return {
        "blocks": blocks,
        "text": result.content,
    }


def confirm_options(
    options: list[str],
    message="Select an option:",
    placeholder="Choose an option...",
    action_id="option_selection",
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

    return {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":thinking_face: *{message}*",
                },
            },
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
        ],
        "text": f"Selection required: {message}",  # Fallback text
    }


def confirm_bool(
    message="Do you want to proceed?",
    yes_text="Yes",
    no_text="No",
) -> dict:
    """Format a confirmation block for Slack with Yes/No buttons and decorative elements.

    Args:
        message: The confirmation question to display
        yes_text: Text for the confirmation button
        no_text: Text for the decline button

    Returns:
        dict: Slack blocks format for a confirmation message

    """
    return {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":question: *{message}*",
                },
            },
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
                            "text": f":x: {no_text}",
                            "emoji": True,
                        },
                        "style": "danger",
                        "value": "cancel",
                        "action_id": "cancel_action",
                    },
                ],
            },
        ],
        "text": f"Confirmation required: {message}",  # Fallback text
    }
