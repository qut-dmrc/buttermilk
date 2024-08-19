import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import local

from azure.identity import DefaultAzureCredential
import cloudpathlib
import pandas as pd
import promptflow as pf
from azure.ai.inference import ChatCompletionsClient
from azure.ai.ml import Input, MLClient, Output, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from jinja2 import Environment, FileSystemLoader
from promptflow._sdk.entities._flows import FlexFlow, Flow
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
from promptflow.core import (
    AzureOpenAIModelConfiguration,
    OpenAIModelConfiguration,
    Prompty,
)
from promptflow.core._connection import (
    AzureOpenAIConnection,
    OpenAIConnection,
    _Connection,
)
from promptflow.tracing import start_trace, trace


from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import CommandComponent, Environment, Code
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
BASE_DIR = Path(__file__).absolute().parent
FLOW_DIR = BASE_DIR / "judge"


from multiprocessing import Process
from pathlib import Path
from tempfile import NamedTemporaryFile
import cloudpathlib
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
from promptflow.tracing import start_trace, trace

from flows.judge.judge import Judger
BASE_DIR = Path(__file__).absolute().parent
FLOW_DIR = BASE_DIR / "judge"
DATASET_DRAG = 'azureml:drag_train:1'


def run() -> None:
    dataset = Input(
        path=DATASET_DRAG,
        type=AssetTypes.URI_FILE,
        mode="mount",
    )

    standards= [('ordinary', 'criteria_ordinary.jinja2'), ('gelber', 'criteria_gelber.jinja2')]
    standards = [('vaw', 'criteria_vaw.jinja2')]
    system_prompt_path= 'instructions.jinja2'
    process_path= 'process.jinja2'
    template_path= 'apply_rules.jinja2'

    credential = DefaultAzureCredential()
    azclient = AzurePFClient.from_config(credential=credential, path="/workspaces/dev/.azureml/config.json")
    mlclient = azclient.ml_client

    localclient = LocalPFClient()
    runs = []

    start_trace()

    try:
        for i, (standards_name, standards_path) in enumerate(standards):
            for j, model in enumerate(CHATMODELS):
                columns = {'source': r'${data.source}',
                        'record_id': r'${data.id}',
                        'content': r'${data.alt_text}',}
                init_vars={
                        'langchain_model_name': model,
                        "standards_path": standards_path,
                        'run_local_models': False,
                        "system_prompt_path": system_prompt_path,
                        "process_path": process_path}
                run_info = {}

                judger = Judger(template_path=template_path,
                                standards_path=standards_path, langchain_model_name=model, system_prompt_path=system_prompt_path, process_path=process_path)

                moderate_run = localclient.run(flow=judger, data=dataset, init_vars=init_vars,column_mapping=columns, stream=False, name=run_name)

                run_info['moderate'] = moderate_run._to_dict(exclude_additional_info=True, exclude_debug_info=True)

                # Execute the eval run
                eval_run_name = f"eval_{moderate_run.name}"
                eval_columns = { "groundtruth": r"${data.expected}", "result": r"${run.outputs.result}"}
                evaluation_run = localclient.run(flow="flows/evaluate", data=dataset, run=moderate_run.name, column_mapping=eval_columns, stream=False, name=eval_run_name)

                run_info['evaluate'] = evaluation_run._to_dict(exclude_additional_info=True, exclude_debug_info=True)

                runs.append(run_info)


if __name__ == "__main__":
    run()

flow_component = load_component("/workspaces/dev/automod/flows/automod/judge/flow.dag.yaml")


def run_component(
    dataset: Input(type="uri_file"),
    output: Output(type="uri_folder")
) -> None:
    # Your existing run() function code goes here
    # Make sure to save outputs to the `output` folder
    pass


def main_pipeline(dataset: Input(type="uri_file")):
    run_job = run_component(dataset=dataset)
    return {"output": run_job.outputs.output}


def submit_pipeline():
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Create a pipeline job
    pipeline_job = main_pipeline(dataset=Input(path="azureml:your_dataset:1"))

    # Submit the job
    returned_job = ml_client.jobs.create_or_update(pipeline_job)

    # Wait for the job to complete
    ml_client.jobs.stream(returned_job.name)

if __name__ == "__main__":
    submit_pipeline()
a