import pprint
import textwrap
from collections.abc import Mapping, Sequence

import regex as re
from pydantic import BaseModel

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
    stripped = ""

    for line in lines:
        text = re.sub(r"\n{2,+}", "\n", line)
        text = str(text) + " "  # add trailing space here, and we'll remove it in the next step if it's duplicated
        text = re.sub(r"\s{2,+}", " ", text)
        stripped += text

    # Break output into a list of strings of max length 3000 characters, wrapping nicely
    return textwrap.wrap(stripped, width=SLACK_MAX_MESSAGE_LENGTH, replace_whitespace=False)  # Preserve newlines


def format_slack_message(result: dict) -> dict:
    """Format message for Slack API with attractive blocks for structured data"""
    blocks = []

    # Add header with model identifier
    model_id = result.get("identifier", result.get("model", "?"))
    header_text = f"Model: :robot_face: {model_id}"
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": header_text,
            "emoji": True,
        },
    })

    # Handle error case
    if result.get("error") and result["error"] not in [None, "None"]:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Error!*\n" + "\n".join(format_response(result.get("error"))),
            },
        })

    else:
        outputs = result.get("outputs", {})

        # Handle feedback metadata fields if present
        feedback_fields = ["mistake", "intervention"]
        if any(result.get(k) for k in feedback_fields):
            feedback_text = ""

            # Format mistake field with special styling
            if "mistake" in result:
                mistake = result.get("mistake")
                icon = ":x:" if mistake and mistake not in [False, "False"] else ":white_check_mark:"
                feedback_text += f"{icon} *Mistake:* {mistake!s}\n\n"

            # Format intervention field
            if "intervention" in result and result.get("intervention"):
                intervention = result.get("intervention")
                feedback_text += f"*Intervention:*\n{intervention}\n"

            if feedback_text:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": feedback_text.strip(),
                    },
                })

                # Add a divider after feedback if we have other content
                if outputs:
                    blocks.append({"type": "divider"})

        if isinstance(outputs, dict):
            # Extract reasons for special handling
            reasons = outputs.pop("reasons", []) if isinstance(outputs, dict) else []

            # Format prediction, confidence and severity with special styling if present
            key_fields = ["prediction", "confidence", "severity"]
            if any(k in outputs for k in key_fields):
                special_text = ""

                if "prediction" in outputs:
                    prediction = outputs.pop("prediction", None)
                    icon = ":white_check_mark:" if prediction else ":no_entry:"
                    special_text += f"{icon} *Prediction:* {prediction!s}\n"

                if "confidence" in outputs:
                    confidence = outputs.pop("confidence", "")
                    special_text += f":bar_chart: *Confidence:* {confidence}\n"

                if "severity" in outputs:
                    severity = outputs.pop("severity", "")
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
            if outputs.get("labels"):
                labels = outputs.pop("labels", [])
                if labels:
                    label_text = "*Labels:* " + " ".join([f"`{label}`" for label in labels])
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": label_text,
                        },
                    })

            # Handle any remaining fields in outputs
            remaining_outputs = {k: v for k, v in outputs.items()
                               if k not in ["reasons", "prediction", "confidence", "severity", "labels"]}

            if remaining_outputs:
                for text in format_response(remaining_outputs):
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": text,
                        },
                    })

            # Add divider before reasons if there are any
            if reasons:
                blocks.append({
                    "type": "divider",
                })

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Reasoning:*",
                    },
                })

                # Add each reason as its own contextual block for better readability
                for i, reason in enumerate(reasons):
                    blocks.append({
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"{i + 1}. {reason}",
                            }
                        ],
                    })
        else:
            # Handle case where outputs is not a dict
            for text in format_response(outputs):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text,
                    },
                })

    # Slack has a limit on blocks, so ensure we don't exceed it
    blocks = blocks[:50]  # Slack's block limit

    # Also provide a text fallback for clients that don't support blocks
    fallback_text = (
        header_text
        + "\n"
        + pprint.pformat(result, indent=2)[
            : SLACK_MAX_MESSAGE_LENGTH - len(header_text) - 10
        ]
    )

    return {
        "blocks": blocks,
        "text": fallback_text,
    }


def confirm_block(
    message="Do you want to proceed?", yes_text="Yes", no_text="No"
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
