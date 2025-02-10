import pytest

from buttermilk._core.agent import Agent
from buttermilk._core.config import SaveInfo
from buttermilk._core.runner_types import Job
from buttermilk.agents.sheetexporter import GSheetExporter


@pytest.fixture
def flow() -> Agent:
    return GSheetExporter(
        save=SaveInfo(type="gsheets", sheet_name="testing_gsheet_export", sheet_id=None)
    )


@pytest.mark.anyio
async def test_gsheet_exporter(flow: Agent):
    job = Job(
        job_id="job_id",
        source="testimng",
        flow_id=flow.name,
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

    assert result.result.sheet_url
