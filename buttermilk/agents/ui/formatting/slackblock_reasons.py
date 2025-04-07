import pprint

from buttermilk._core.contract import AgentOutput
from buttermilk._core.defaults import SLACK_MAX_MESSAGE_LENGTH
from buttermilk.agents.ui.formatting.slackblock import format_response


def format_slack_reasons(result: AgentOutput) -> dict:
    """Format message for Slack API with attractive blocks for structured data"""
    blocks = []
    result_copy = result.model_copy(deep=True)

    # Add header with model identifier
    header_text = f"Model: :robot_face: {result_copy.role} {result_copy.metadata.get('model')}"
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": header_text,
            "emoji": True,
        },
    })
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": ", ".join(result_copy.inputs.values()),
        },
    })

    # Handle error case
    if result_copy.outputs.get("error"):
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Error!*\n"
                + "\n".join(format_response(result_copy.outputs.get("error"))),
            },
        })

    else:
        outputs = result_copy.outputs
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
                            },
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
        + pprint.pformat(result_copy, indent=2)[
            : SLACK_MAX_MESSAGE_LENGTH - len(header_text) - 10
        ]
    )

    return {
        "blocks": blocks,
        "text": fallback_text,
    }
