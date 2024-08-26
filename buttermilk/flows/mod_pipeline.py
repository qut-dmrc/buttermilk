
from multiprocessing import Process
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
from promptflow.tracing import start_trace, trace


from azure.ai.ml import dsl, Input, Output
from azure.identity import DefaultAzureCredential

from buttermilk.utils.utils import make_run_id
BASE_DIR = Path(__file__).absolute().parent
FLOW_DIR = BASE_DIR / "judge"


BASE_DIR = Path(__file__).absolute().parent
FLOW_DIR = BASE_DIR / "judge"
DATASET_DRAG = 'azureml:drag_train:1'


def run_local(flow_folder, flow_input_data, run_name, model):
    standards_path = 'criteria_ordinary.jinja2'
    system_prompt_path = 'instructions.jinja2'
    process_path = 'process.jinja2'
    template_path = 'apply_rules.jinja2'
    columns = {'record_id': r'${data.id}',
               'content': r'${data.alt_text}',
               'langchain_model_name': model,
               "standards_path": standards_path,
               "template_path": template_path,
               "system_prompt_path": system_prompt_path,
               "process_path": process_path}
    column_mapping = " ".join([f"{k}='{v}'" for k, v in columns.items()])
    cmd = (
        f'pf run create --flow "{flow_folder}" --data "{flow_input_data}" --name {run_name} '
        f"--environment-variables PF_WORKER_COUNT=20 PF_BATCH_METHOD='spawn' "
        f"--column-mapping {column_mapping} --debug"
    )
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # f"Submit batch run successfully. process id {process.pid}. Please wait for the batch run to complete..."
    print(f"Run cmd: {cmd}")

#            "azureml://subscriptions/7e7e056a-4224-4e26-99d2-1e3f9a688c50/resourcegroups/rg-suzor_ai/workspaces/automod/datastores/workspaceblobstore/paths/UI/2024-08-19_091358_UTC/drag_train.jsonl"

def run() -> None:
    dataset = Input(
        path="azureml://subscriptions/7e7e056a-4224-4e26-99d2-1e3f9a688c50/resourcegroups/rg-suzor_ai/workspaces/automod/datastores/workspaceblobstore/paths/UI/2024-08-14_081226_UTC/drag_train.jsonl",
        type=AssetTypes.URI_FILE,
        mode="mount",
    )

    run_id = make_run_id()
    # run_local(flow_folder="flows/apply", flow_input_data=dataset,
    #           run_name=run_id, model="claude35sonnet")=
    credential = DefaultAzureCredential()
    azclient = AzurePFClient.from_config(
        credential=credential, path="/workspaces/dev/.azureml/config.json")
    mlclient = azclient.ml_client

    localclient = LocalPFClient()
    runs = []

    start_trace()
    standards_path = 'criteria_ordinary.jinja2'
    system_prompt_path = 'instructions.jinja2'
    process_path = 'process.jinja2'
    template_path = 'apply_rules.jinja2'
    columns = {'record_id': r'${data.id}',
               'content': r'${data.alt_text}'}
    init_vars = {
               'langchain_model_name': "claude35sonnet",
               "standards_path": standards_path,
               "template_path": template_path,
               "system_prompt_path": system_prompt_path,
               "process_path": process_path}
    job = load_component(
        "flows/apply/flow.dag.yaml", params_override=[init_vars])(
            data=dataset, **columns)

    # Submit the job
    returned_job = mlclient.jobs.create_or_update(job)

    # Wait for the job to complete
    mlclient.jobs.stream(returned_job.name)

    run_job = run_component(dataset=dataset)
    result = {"output": run_job.outputs.output}

    # Execute the eval run
    eval_run_name = f"eval_{moderate_run.name}"
    eval_columns = {"groundtruth": r"${data.expected}",
                    "result": r"${run.outputs.result}"}
    evaluation_run = localclient.run(flow="flows/evaluate", data=dataset, run=moderate_run.name,
                                     column_mapping=eval_columns, stream=False, name=eval_run_name)

    run_info['evaluate'] = evaluation_run._to_dict(
        exclude_additional_info=True, exclude_debug_info=True)

    runs.append(run_info)


if __name__ == "__main__":
    run()
