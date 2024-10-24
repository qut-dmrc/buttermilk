import datetime
import time
from pathlib import Path
from typing import Optional, Self, TypedDict

import pandas as pd
import tqdm
from jinja2 import BaseLoader, Environment, FileSystemLoader, Undefined
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from promptflow.core import Prompty, ToolProvider, tool
from promptflow.core._prompty_utils import convert_prompt_template

from buttermilk import BM
from buttermilk.flows.common.config import COL_PREDICTION
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser

BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATHS = [BASE_DIR, BASE_DIR.parent / "common", BASE_DIR.parent / "templates"]


class Prediction(TypedDict):
    timestamp: str
    correct: bool
    alignment: float
    reasons: dict
    groundtruth: dict


class KeepUndefined(Undefined):
    def __str__(self):
        return "{{ " + self._undefined_name + " }}"


class EvalQA(ToolProvider):
    def __init__(self, *, model: str, template_path: str = "evalqa.jinja2") -> None:
        bm = BM()
        self.connections = bm._connections_azure

        env = Environment(loader=FileSystemLoader(searchpath=TEMPLATE_PATHS), trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        tpl = env.get_template(template_path).render({})
        template_vars = {"groundtruth": "{{groundtruth}}", COL_PREDICTION:r"{{" + COL_PREDICTION + "}}", "reasons":r"{{reasons}}"}
        template = convert_prompt_template(tpl, api="chat", inputs=template_vars)

        # convert to a list of messages and roles expected by langchain
        langchain_messages = [
            (m["role"], m["content"]) for m in template if m["content"]
        ]

        self.template = ChatPromptTemplate.from_messages(
            langchain_messages, template_format="jinja2"
        )

        self.model = model

    @tool
    def __call__(self, *, groundtruth: dict, scored_result: dict, reasons: dict, **kwargs) -> Prediction:

        prediction=scored_result['prediction']

        llm = LLMs(connections=self.connections)[self.model]

        chain = self.template.copy() | llm | ChatParser()
        input_vars = dict(
            groundtruth=groundtruth,
            prediction=prediction,
            reasons=reasons,
        )
        output = chain.invoke(input=input_vars)
        output['result'] = scored_result
        output['groundtruth'] = groundtruth
        output["timestamp"] = pd.to_datetime(datetime.datetime.now(tz=datetime.UTC)).isoformat()
        for k in Prediction.__required_keys__:
            if k not in output:
                output[k] = None

        return output

    def batch(self, dataset: pd.DataFrame, groundtruth: str = 'expected', prediction: str = 'prediction', **kwargs) -> list:
        results = []
        for _, row in tqdm.tqdm(
            dataset.iterrows(),
            total=dataset.shape[0],
            desc=f'evaluator-{self.model}',
            bar_format="{desc:30}: {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            details = self.__call__(groundtruth=row[groundtruth], prediction=row[prediction], **kwargs)

            results.append(details)

        return results