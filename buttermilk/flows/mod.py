import datetime
from multiprocessing import Process
from pathlib import Path
from random import shuffle
import resource
from tempfile import NamedTemporaryFile
import cloudpathlib
import pandas as pd
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
from promptflow.tracing import start_trace, trace

from buttermilk.utils import make_run_id, read_json
from buttermilk.flows.apply.judge import Judger
from datatools.chains.llm import CHATMODELS, LLMs
from datatools.gcloud import GCloud
from datatools.log import getLogger
from datatools.azcloud import auth

BASE_DIR = Path(__file__).absolute().parent
FLOW_DIR = BASE_DIR / "judge"


logger = getLogger()

BASE_DIR = Path(__file__).absolute().parent

DATASET_OSB = "gs://dmrc-platforms/data/osb_train.jsonl"
DATASET_DRAG = "gs://dmrc-platforms/data/drag_train.jsonl"
DATASET_TONEPOLICE = "gs://dmrc-platforms/data/tonepolice_test.jsonl"
DATASET_VAW = "gs://dmrc-platforms/data/vaw_train.jsonl"

def run_batch(gc, standards_name, standards_path, model, system_prompt_path, process_path, template_path, run_id, dataset, bq_results, rules_schema, metrics_schema, bq_metrics ) -> None:

    localclient = LocalPFClient()
    logger = gc.logger

    columns = {
                    "record_id": r"${data.id}",
                    "content": r"${data.alt_text}",
                }
    init_vars = {
        "langchain_model_name": model,
        "standards_path": standards_path,
        "system_prompt_path": system_prompt_path,
        "process_path": process_path,
        "template_path": template_path,
    }

    run_name = f"{run_id}_{standards_name}_{model}"
    run_time = datetime.datetime.now().isoformat()
    run_meta = {
        "init": init_vars,
        "standards": standards_name,
        "model": model,
    }
    # Execute the run
    logger.info(
        f"Starting run '{run_name}' in Azure ML. This can take time.",
    )

    judger = Judger(**init_vars)

    moderate_run = localclient.run(
        flow=judger,
        data=dataset,
        init_vars=init_vars,
        column_mapping=columns,
        stream=False,
        name=run_name,display_name="Automod",timeout=150,
    )

    details = localclient.get_details(moderate_run.name)

    details = details.rename(columns={"inputs.line_number":"line_number"})

    logger.info(
        f"Run {moderate_run.name} completed with status {moderate_run.status}. URL: {moderate_run._portal_url}."
    )
    run_meta["moderation_run_name"] = moderate_run.name

    if moderate_run.status != "Completed":
        logger.error(
            f"Run {moderate_run.name} did not complete successfully. Skipping evaluation."
        )
        if details.shape[0] == 0:
            logger.error(f"No rows processed in run {moderate_run.name}.")
            return
        # duplicate run_meta for each row
        details.loc[:, "run_info"] = [run_meta for _ in range(details.shape[0])]
        results_uri = gc.save(data=details, schema=rules_schema, dataset=bq_results)
        logger.info(f"Failed run results saved to {results_uri}, metrics not run.")
        return

    # stack inputs into dicts
    cols = [c for c in details.columns if c.lower().startswith("inputs.")]
    df_tmp = details[cols]
    df_tmp.columns = [c.replace("inputs.", "") for c in cols]
    details.loc[:, "inputs"] = details[cols].apply(dict, axis=1)
    details = details.drop(columns=cols)

    details.loc[:, "run_name"] = run_name
    details.loc[:, "timestamp"] = run_time

    # Execute the eval run
    eval_run_name = f"eval_{moderate_run.name}"
    logger.info(
        f"Starting evaluation run '{eval_run_name}' in Azure ML. This can take time.",
    )
    eval_columns = {
        "record_id": r"${data.id}",
        "groundtruth": r"${data.expected}",
        "result": r"${run.outputs.result}",
    }
    evaluation_run = localclient.run(
        flow="flows/evaluate",
        data=dataset,
        run=moderate_run.name,
        column_mapping=eval_columns,
        stream=False,
        name=eval_run_name,display_name="Automod",timeout=150,
    )

    logger.info(
        f"Evaluation run {evaluation_run.name} completed with status {evaluation_run.status}. URL: {evaluation_run._portal_url}."
    )
    evals = localclient.get_details(evaluation_run.name)
    evals = evals.drop(columns=["inputs.result", "inputs.groundtruth"])

    evals = evals.rename(
        columns={"outputs.summary": "summary", "outputs.result": "result", "inputs.line_number":"line_number"}
    )

    details = details.merge(evals, how="outer", on="line_number")

    run_meta["eval_run_name"] = evaluation_run.name

    # duplicate run_meta for each row
    details.loc[:, "run_info"] = [run_meta for _ in range(details.shape[0])]

    results_uri = gc.save(data=details, schema=rules_schema, dataset=bq_results)

    metrics = localclient.get_metrics(evaluation_run.name)
    metrics_info = {}
    metrics_info["run_name"] = run_name
    metrics_info["timestamp"] = run_time
    metrics_info["metrics"] = metrics
    metrics_info["run_info"] = run_meta.copy()
    metrics_info["moderate"] = moderate_run._to_dict(
        exclude_additional_info=True, exclude_debug_info=True
    )
    metrics_info["evaluate"] = evaluation_run._to_dict(
        exclude_additional_info=True, exclude_debug_info=True
    )
    metrics_uri = gc.save(data=[metrics_info], schema=metrics_schema, dataset=bq_metrics)
    logger.info(f"Run results saved to {results_uri}, metrics saved to {metrics_uri}")

def run() -> None:
    gc = GCloud(name="automod", job="drag")
    logger = gc.logger
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as f:
        dataset = f.name
    cloudpathlib.CloudPath(DATASET_DRAG).download_to(dataset)
    bq_results = "dmrc-analysis.toxicity.standards"
    rules_schema = read_json("flows/common/rules.schema.json")
    metrics_schema= read_json("flows/common/metrics.schema.json")
    bq_metrics = "dmrc-analysis.toxicity.metrics"
    standards = [
        ("ordinary", "criteria_ordinary.jinja2"),
        ("gelber", "criteria_gelber.jinja2"),
    ]
    # standards = [('vaw', 'criteria_vaw.jinja2')]
    system_prompt_path = "instructions.jinja2"
    process_path = "process.jinja2"
    template_path = "apply_rules.jinja2"

    run_id = make_run_id()
    start_trace(resource_attributes={"run_id": run_id}, collection="automod")
    shuffle(CHATMODELS)
    for i, (standards_name, standards_path) in enumerate(standards):
        for j, model in enumerate(CHATMODELS):
            try:
                run_batch(gc, standards_name, standards_path, model, system_prompt_path, process_path, template_path, run_id, dataset, bq_results, rules_schema, metrics_schema, bq_metrics)
            except Exception as e:
                logger.error(f"Unhandled error in our flow: {e}")

    Path(dataset).unlink(missing_ok=True)


if __name__ == "__main__":
    run()
