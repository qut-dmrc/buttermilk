
from multiprocessing import Process
from pathlib import Path
from tempfile import NamedTemporaryFile
import cloudpathlib
import pandas as pd
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
from promptflow.tracing import start_trace, trace

from buttermilk.utils import make_run_id
from flows.judge.judge import Judger
from datatools.chains.llm import CHATMODELS, LLMs
from datatools.gcloud import GCloud
from datatools.log import getLogger
from datatools.azcloud import auth
BASE_DIR = Path(__file__).absolute().parent
FLOW_DIR = BASE_DIR / "judge"


logger = getLogger()

BASE_DIR = Path(__file__).absolute().parent

DATASET_OSB = 'gs://dmrc-platforms/data/osb_train.jsonl'
DATASET_DRAG = 'gs://dmrc-platforms/data/drag_train.jsonl'
DATASET_TONEPOLICE = 'gs://dmrc-platforms/data/tonepolice_test.jsonl'
DATASET_VAW = 'gs://dmrc-platforms/data/vaw_train.jsonl'

def run() -> None:
    gc=GCloud(name="automod", job="drag")
    logger=gc.logger
    with NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w') as f:
        dataset = f.name
    cloudpathlib.CloudPath(DATASET_DRAG).download_to(dataset)

    # dataset = Input(
    #     path="azureml:osb_train:1",
    #     type=AssetTypes.URI_FILE,
    #     mode="mount",
    # )

    standards= [('ordinary', 'criteria_ordinary.jinja2'), ('gelber', 'criteria_gelber.jinja2')]
    #standards = [('vaw', 'criteria_vaw.jinja2')]
    system_prompt_path= 'instructions.jinja2'
    process_path= 'process.jinja2'
    template_path= 'apply_rules.jinja2'

    credential = auth()
    azclient = AzurePFClient.from_config(credential=credential, path="/workspaces/dev/.azureml/config.json")
    mlclient = azclient.ml_client
    run_id = make_run_id()

    localclient = LocalPFClient()
    runs = []
    results = pd.DataFrame()

    start_trace()

    judger = None
    try:
        for i, (standards_name, standards_path) in enumerate(standards):
            for j, model in enumerate(CHATMODELS):
                columns = { 'record_id': r'${data.id}',
                        'content': r'${data.alt_text}',}
                init_vars={
                        'langchain_model_name': model,
                        "standards_path": standards_path,
                        'run_local_models': False,
                        "system_prompt_path": system_prompt_path,
                        "process_path": process_path}
                run_name = f"{run_id}_{standards_name}_{model}"
                run_info = {}
                run_info["run_name"] = run_name

                # Execute the run
                logger.info(
                    f"Starting run '{run_name}' in Azure ML. This can take time.",
                )

                judger = Judger(template_path=template_path,
                                standards_path=standards_path, langchain_model_name=model, system_prompt_path=system_prompt_path, process_path=process_path)

                moderate_run = localclient.run(flow=judger, data=dataset, init_vars=init_vars,column_mapping=columns, stream=False, name=run_name)

                run_info['moderate'] = moderate_run._to_dict(exclude_additional_info=True, exclude_debug_info=True)
                details = localclient.get_details(moderate_run.name)

                logger.info(f"Run {moderate_run.name} completed with status {moderate_run.status}. URL: {moderate_run._portal_url}.")

                # Execute the eval run
                eval_run_name = f"eval_{moderate_run.name}"
                logger.info(
                    f"Starting evaluation run '{eval_run_name}' in Azure ML. This can take time.",
                )
                eval_columns = { "groundtruth": r"${data.expected}", "result": r"${run.outputs.result}"}
                evaluation_run = localclient.run(flow="flows/evaluate", data=dataset, run=moderate_run.name, column_mapping=eval_columns, stream=False, name=eval_run_name)

                run_info['evaluate'] = evaluation_run._to_dict(exclude_additional_info=True, exclude_debug_info=True)
                logger.info(f"Evaluation run {evaluation_run.name} completed with status {evaluation_run.status}. URL: {evaluation_run._portal_url}.")
                evals = localclient.get_details(evaluation_run.name)
                details = details.merge(evals, how='outer', on='inputs.line_number')
                details.loc[:, 'run_name'] = run_name
                details.loc[:, 'standards'] = standards_name
                details.loc[:, 'model'] = model
                details.loc[:, 'eval_run_name'] = evaluation_run.name
                details.loc[:, 'moderation_run_name'] = moderate_run.name
                runs.append(run_info)
                results = pd.concat([results, details], ignore_index=True)
    except Exception as e:
        logger.error(f"Unhandled error in our flow: {e}")
        raise e
    finally:
        uri = gc.save(data=runs)
        logger.info(f"Runs saved to {uri}")


if __name__ == "__main__":
    run()