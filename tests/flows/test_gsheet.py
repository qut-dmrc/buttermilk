import pytest

from buttermilk._core.agent import Agent
from buttermilk._core.config import SaveInfo

from buttermilk.agents.sheetexporter import GSheetExporter


@pytest.fixture
def flow() -> Agent:
    return GSheetExporter(
        save=SaveInfo(
            type="gsheets", sheet_name="testing_gsheet_export", sheet_id=None
        ),
    )


## Flow config
export_config = {
    "convert_json_columns": [
        "record",
        "owl",
        "reasons",
        "differences"
    ],
    "save": {
        "type": "gsheets",
        "sheet_id": None,
        "sheet_name": "testing",
        "title": "testing only"
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
        "differences": "eval.differences"
    }
}

@pytest.mark.anyio
async def test_gsheet_exporter(flow: Agent):
    job = Job(
        job_id="job_id",
        source="testing",
        flow_id=flow.agent_id,
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
    )
    result = await flow.process_job(job=job)

    assert result.outputs['sheet_url']
