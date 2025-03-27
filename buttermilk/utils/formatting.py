import json

from rich.console import Console


def json_to_slack_format(data):
    """Convert JSON data to Slack-friendly format."""
    # Create an in-memory console
    console = Console(record=True, width=80)

    # Use Rich to print the JSON
    console.print_json(json.dumps(data))

    # Get the Markdown version (works better with Slack than ANSI)
    md_output = console.export_markdown()

    # Clean up the output for Slack
    # Strip unnecessary backticks that Slack might not render correctly
    return md_output.replace("```json\n", "```\n")
