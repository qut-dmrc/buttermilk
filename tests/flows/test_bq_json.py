# This test addresses issue #14 https://github.com/qut-dmrc/buttermilk/issues/14
# results saved to bq JSON field as string not object.
# e.g.output from a flow run, each row is a string, not a dict

import json

import numpy as np
import pandas as pd
import pytest

from buttermilk._core.agent import save_job
from buttermilk._core.runner_types import Job
from buttermilk.utils.save import data_to_export_rows

SAMPLE_DATA = [
    {
        "labels": [],
        "reasons": [
            "The reasoning provided generally aligns with the expected key points regarding the article's coverage of a legal case involving a transgender former Kansas Highway Patrol employee.",
            "The analysis correctly notes that the article carefully navigates pronoun usage and acknowledges uncertainty about the individual's current preferences.",
            "The reasoning also acknowledges potential misgendering and deadnaming due to uncertainty about current preferences.",
            "The analyst correctly evaluates the article's focus on the relevant aspects of the case without sensationalizing the individual's transgender status.",
            "However, the reasoning could be more comprehensive in addressing the gap in abiding by best practice guidelines of respecting current names and pronouns.",
        ],
        "score": 0.85,
    },
    {
        "labels": [],
        "reasons": [
            "The analysis lacks a clear assessment of the article's alignment with guidelines for respectful coverage of transgender individuals, focusing instead on the specifics of the lawsuit and settlement.",
            "While the analysis notes the absence of overtly offensive language, it does not thoroughly examine the article's use of pronouns or naming conventions in relation to best practices for respecting current names and pronouns.",
            "The analysis highlights the potential for perpetuation of negative stereotypes through the reportage of allegations, but does not delve deeper into the implications of presenting such claims without more thorough critique.",
            "The analysis does note the neutral tone of the article, but fails to comprehensively evaluate the article's strategy in balancing the need for factual reporting with sensitivity towards the individual's transgender status.",
            "Overall, the analysis lacks depth in assessing the article's adherence to best practices in covering transgender issues, focusing more on the factual aspects of the case.",
        ],
        "score": 0.5,
    },
    {
        "labels": [],
        "reasons": [
            "The analysis correctly identifies the respectful handling of names by using only 'Dawes' and avoiding potential deadnaming issues, which aligns with expected good practices",
            "The evaluation appropriately notes the problematic use of 'him' pronouns from court documents without clarification of current preferences, matching our concern about pronoun usage",
            "The analysis recognizes the article's factual focus and legal context, particularly noting the connection to Supreme Court protections for LGBTQ+ individuals",
            "The assessment correctly identifies the lack of direct representation through quotes from Dawes or their attorney as a limitation, showing awareness of trans-centered reporting guidelines",
            "The response demonstrates understanding of best practices in trans coverage while acknowledging the practical constraints faced in this specific case due to limited information",
        ],
        "score": 0.85,
    },
]


@pytest.fixture
def joblist(job_minimal: Job):
    data = []
    for line in SAMPLE_DATA:
        job = job_minimal.model_copy()
        job.outputs = line
        data.append(job)

    return data


def test_issue_14_job(joblist):
    for job in joblist:
        assert isinstance(job.outputs, dict)
        assert len(job.outputs["reasons"]) == 5
        assert job.outputs["labels"] == []
    assert joblist[1].outputs.score == 0.5
    assert joblist[2].outputs.score == 0.85
    assert joblist[2].outputs["reasons"][4].startswith("The response demonstrates")


def test_issue_14_rows(joblist, bm, flow):
    schema = flow.steps[0].save.db_schema

    if isinstance(schema, str):
        schema = bm.bq.schema_from_json(schema)

    bq_rows = [x for job in joblist for x in data_to_export_rows(job, schema=schema)]

    for row in bq_rows:
        assert len(row["outputs"]["reasons"]) == 5
        assert row["outputs"]["labels"] == []
    assert bq_rows[1]["outputs"]["score"] == 0.5
    assert bq_rows[2]["outputs"]["score"] == 0.85
    assert bq_rows[2]["outputs"]["reasons"][4].startswith("The response demonstrates")


def test_issue_14_df(joblist, bm, flow):
    df = pd.DataFrame([job.model_dump() for job in joblist])

    schema = flow.steps[0].save.db_schema

    if isinstance(schema, str):
        schema = bm.bq.schema_from_json(schema)

    bq_rows = data_to_export_rows(df, schema=schema)

    for row in bq_rows:
        assert len(row["outputs"]["reasons"]) == 5
        assert row["outputs"]["labels"] == []
    assert bq_rows[1]["outputs"]["score"] == 0.5
    assert bq_rows[2]["outputs"]["score"] == 0.85
    assert bq_rows[2]["outputs"]["reasons"][4].startswith("The response demonstrates")


def test_issue_14_upload(joblist, flow, bm):
    save_info = flow.steps[0].save
    job_ids = ", ".join([f"'{job.job_id}'" for job in joblist])

    for job in joblist:
        destination = save_job(job, save_info=save_info)
        assert destination == save_info.dataset

    sql = f"SELECT outputs.score, outputs.reasons, outputs.labels FROM {save_info.dataset} WHERE job_id IN ({job_ids}) AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 300 SECOND)"
    df = bm.run_query(sql)
    assert np.allclose(
        df["score"].to_numpy().astype(float),
        np.array([0.85, 0.5, 0.85]),
        rtol=1e-15,
        atol=0,
    )
    assert not df["labels"].apply(json.loads).any()

    # best we can do at the moment i think
    reason = json.loads(df["reasons"].values[2])[4]
    assert reason.startswith("The response demonstrates")

    # this should ideally work but we need to convert the whole hierarchy first...
    # assert df["reasons"][2][4].startswith("The response demonstrates")
