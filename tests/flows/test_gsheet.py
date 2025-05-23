import pytest

from buttermilk._core.contract import AgentInput  # Import AgentInput
from buttermilk.agents.sheetexporter import GSheetExporter


@pytest.fixture
def flow() -> GSheetExporter: # Changed type hint to GSheetExporter
    return GSheetExporter(
        session_id="test_session", # Required by Agent
        role="EXPORTER", # Required by AgentConfig
        parameters={ # Pass agent-specific config in parameters
            "save": { # SaveInfo config goes inside parameters
                "type": "gsheets",
                "sheet_name": "testing_gsheet_export", # Added colon
                "sheet_id": None,
                "db_schema": "some_schema.json", # Added colon
            },
            # Add other parameters if needed by GSheetExporter
        }
        # Add other required AgentConfig fields if any (like description)
    )


# Flow config
export_config = {
    "convert_json_columns": [
        "record",
        "owl",
        "reasons",
        "differences",
    ],
    "save": {
        "type": "gsheets",
        "sheet_id": None,
        "sheet_name": "testing",
        "title": "testing only",
    },
    "inputs": {
        "flow_id": "flow_id",
        "record_id": "record.record_id",
        "job_id": "job_id",
        "timestamp": "timestamp",
        "record": "record.text",
        "owl": "context_owl",
        "scores": "scorer.score",
        "reasons": "scorer",
        "differences": "eval.differences",
    },
}


@pytest.mark.anyio
async def test_gsheet_exporter(flow: GSheetExporter): # Changed type hint to GSheetExporter
    agent_input = AgentInput( # Replaced Job with AgentInput
        parameters={
            "sheet_name": "evals",
            "title": "Trans Guidelines Judger",
        },
        inputs={
            "job_id": "job_id",
            "timestamp": "timestamp",
            "title": "Trans Guidelines Judger",
            "prompt": "Please evaluate the following translation based on the given criteria: {criteria}.",
            "criteria": "The translation should be grammatically correct and accurate.",
            "model": "gpt-3.5-turbo",
            "answer": "The translation is grammatically correct and accurate.",
            "record": "record",
            "answers": "judger.answers",
            "synthesis": "synth.answers",
            "differences": "eval.analysis",
            "model": "eval.model",
        },
        # Removed session_id from AgentInput
    )
    result = await flow.process_job(message=agent_input) # Pass agent_input

    assert result.outputs["sheet_url"]
