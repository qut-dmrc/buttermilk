""" An Azure PromptFlow flavoured wrapper around a Langchain model call."""
import datetime
import sys
import time
from cgitb import text
from logging import getLogger
from pathlib import Path
from typing import Optional, Self, TypedDict

import pandas as pd
from jinja2 import BaseLoader, Environment, FileSystemLoader, Undefined
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from promptflow.core import ToolProvider, tool
from promptflow.tracing import trace

from buttermilk import BM
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser

BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATH = BASE_DIR.parent / "templates"

logger = getLogger()

class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'

class Prediction(TypedDict):
    timestamp: object
    metadata: dict
    record_id: str
    analysis: str

def prompt_with_media(
    *, uri=None, mime_type: Optional[str] = None, image_b64: Optional[str] = None, text: Optional[str] = None, detail="auto"
) -> HumanMessage:
    # Prepare input for model consumption


    text_message = {
        "type": "text",
        "text": text or '(see attached)',
    }

    if uri and not mime_type:
        media_message = {
        "type": "image_url",
        "image_url": {
            "url": uri,
        },
        }

    elif uri:
        media_message = {"type": "media", 'mime_type': 'video/mp4',
                              'file_uri': uri}
    elif image_b64:
        media_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": detail,
                },
            }
    else:
        return HumanMessage(content=text or '')

    message = HumanMessage(content=[media_message, text_message])
    return message

class Analyst():
    def __init__(self, *, model: str, template: str = '', system_prompt: str = '', user_prompt: str = '', **kwargs) -> None:
        bm = BM()
        self.tpl_variables = kwargs
        self.model = model
        self.connections = bm._connections_azure

        if template:
            self.tpl_variables['system'] = system_prompt
            self.tpl_variables['user'] = user_prompt

            # Load the template from a prompty file
            from promptflow.core import Prompty
            from promptflow.core._prompty_utils import convert_prompt_template

            # load prompty as a flow
            prompty = Prompty.load(BASE_DIR / template)
            template = convert_prompt_template(prompty._template, api="chat", inputs=self.tpl_variables)

            # convert to a list of messages and roles expected by langchain
            self.langchain_template = [(m['role'], m['content']) for m in template if m['content']]

        elif system_prompt:
            self.langchain_template = [('system', system_prompt)]
            if user_prompt:
                self.langchain_template.append(('user', user_prompt))
        else:
            raise ValueError("Either prompt_template_path or system_prompt must be provided")

        pass


    @tool
    def __call__(
        self, *, content: str, media_attachment_uri=None, record_id: Optional[str] = None, **kwargs) -> Prediction:
        # Add a human message to the end of the prompt
        messages = self.langchain_template + [prompt_with_media(uri=media_attachment_uri, text=content)]

        llm = LLMs(connections=self.connections)[self.model]
        tpl = ChatPromptTemplate.from_messages(messages, template_format="jinja2")

        chain = tpl | llm | ChatParser()
        input_vars = dict()
        input_vars.update({k: v for k, v in kwargs.items() if v})

        output = self.invoke_langchain(chain=chain, input_vars=input_vars, **kwargs)

        if record_id:
            output['record_id'] = record_id

        output["timestamp"] = pd.to_datetime(datetime.datetime.now(tz=datetime.UTC)).isoformat()

        for k in Prediction.__required_keys__:
            if k not in output:
                output[k] = None
        return  output

    @trace
    def invoke_langchain(self, *, chain, input_vars, **kwargs):
        t0 = time.time()
        response = chain.invoke(input=input_vars, **kwargs)
        t1 = time.time()
        logger.debug(f"Finished chain with {self.model} in {t1-t0:.2f} seconds.")
        return response