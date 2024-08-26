
from logging import getLogger
from pathlib import Path
import time

from promptflow.core import (
    ToolProvider,
    tool,
)
from promptflow.tracing import trace
from pathlib import Path
from typing import Optional, Self, TypedDict
from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from jinja2 import Environment, BaseLoader, Undefined
from langchain_core.messages import SystemMessage
from buttermilk import BM
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser

BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATH = BASE_DIR.parent / "templates"

logger = getLogger()

class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'

class LLMOutput(TypedDict):
    metadata: dict
    record_id: str
    analysis: str


def prompt_with_media(
    *, uri=None, image_b64: Optional[str] = None, text: Optional[str] = None, detail="auto"
) -> HumanMessage:
    text_message = {"type": "text", "text": text}

    if uri:
        media_message = {
                        "type": "image_url",
                        "image_url": {
                            "url": uri,
                        },
                    }
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

    return HumanMessage(content=[media_message, text_message])

class Analyst():
    def __init__(self, *, langchain_model_name: str, prompt_template_path: str, system_prompt: str = '', instructions: str = '', output_format: str='',  **kwargs) -> None:

        bm = BM()
        self.connections = bm._connections_azure

        self.langchain_model_name = langchain_model_name
        self.metadata = kwargs

        # This is a bit of a hack to allow Prompty to interpret the template first, but then
        # leave the actual input variables for us to fill later with langchain.
        self.tpl_variables = {"content":r"{{content}}"}
        if system_prompt or instructions or output_format:
            # Load sub-templates ourselves, since Promptflow's prompty won't do it for us.
            env = Environment(loader=FileSystemLoader(searchpath=[BASE_DIR, TEMPLATE_PATH]), trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)
            system_tpl = env.get_template(system_prompt).render()
            instructions_tpl = env.get_template(instructions).render()
            output_format_tpl = env.get_template(output_format).render()

            self.tpl_variables.update(dict(system_prompt=system_tpl, instructions=instructions_tpl, output_format=output_format_tpl))

        # Load the template from a prompty file
        from promptflow.core import Prompty
        from promptflow.core._prompty_utils import convert_prompt_template

        # load prompty as a flow
        prompty = Prompty.load(BASE_DIR / prompt_template_path)
        template = convert_prompt_template(prompty._template, api="chat", inputs=self.tpl_variables)

        # convert to a list of messages and roles expected by langchain
        self.langchain_template = [(m['role'], m['content']) for m in template]
        pass


    @tool
    @trace
    def __call__(
        self, *, content: str, media_attachment_uri=None, record_id: str = 'not given', **kwargs) -> LLMOutput:
        llm = LLMs(connections=self.connections)[self.langchain_model_name]

        if media_attachment_uri:
            message = self.langchain_template[-1][1]
            message = prompt_with_media(uri=media_attachment_uri, text=message)
            messages = self.langchain_template[:-1] + [message]
            tpl = ChatPromptTemplate.from_messages(messages, template_format="jinja2")
        else:
            tpl = ChatPromptTemplate.from_messages(self.langchain_template, template_format="jinja2")

        chain = tpl | llm | ChatParser()
        input_vars = dict(content=content)
        input_vars.update({k: v for k, v in kwargs.items() if v})

        logger.info(f"Invoking chain with {self.langchain_model_name} for record: {record_id}")
        t0 = time.time()
        output = self.invoke_langchain(chain=chain, input_vars=input_vars, **kwargs)
        t1 = time.time()
        logger.info(f"Invoked chain with {self.langchain_model_name} and record: {record_id} in {t1-t0:.2f} seconds")

        output['record_id'] = record_id
        for k in LLMOutput.__required_keys__:
            if k not in output:
                output[k] = None
        output["metadata"] = self.metadata
        return  output

    @trace
    def invoke_langchain(self, *, chain, input_vars, **kwargs):
        return chain.invoke(input=input_vars, **kwargs)