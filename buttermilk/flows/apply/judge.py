
from pathlib import Path

from promptflow.core import (
    ToolProvider,
    tool,
)
from pathlib import Path
from typing import Optional, Self, TypedDict
from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from datatools.chains.llm import LLMs
from datatools.chains.parser import ChatParser
from langchain_core.runnables import Runnable
from jinja2 import Environment, BaseLoader, Undefined

BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATHS = [BASE_DIR, BASE_DIR.parent / "common"]


class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'

class LLMOutput(TypedDict):
    result: dict
    reasons: list
    labels: list
    metadata: dict
    record_id: str

class Judger():

    def template_from_files(self,*, template_path, standards_path,system_prompt_path,process_path) -> ChatPromptTemplate:

        env = Environment(loader=FileSystemLoader(searchpath=TEMPLATE_PATHS), trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)
        criteria = env.get_template(standards_path).render()
        system_prompt = env.get_template(system_prompt_path).render()
        process = env.get_template(process_path).render()

        partial_variables=dict(criteria=criteria, system_prompt=system_prompt, process=process)

        user_prompt = env.get_template(template_path).render(**partial_variables)

        template = ChatPromptTemplate.from_messages([("system",system_prompt), ("human",user_prompt)], template_format="jinja2")


        return template


    def __init__(self, *, langchain_model_name: str,
                 system_prompt: str = None, user_prompt: str =None, standards_path: str = None, template_path: str = None, system_prompt_path: str = None, process_path: str =None) -> None:

        self.langchain_model_name = langchain_model_name

        if standards_path:
            self.template = self.template_from_files(template_path=template_path, standards_path=standards_path, system_prompt_path=system_prompt_path, process_path=process_path)
        else:
            self.template = ChatPromptTemplate.from_messages([("system",system_prompt), ("human",user_prompt)], template_format="jinja2")


    @tool
    def __call__(
        self, *, content: str,**kwargs) -> LLMOutput:

        llm = LLMs()[self.langchain_model_name]

        chain = self.template.copy() | llm | ChatParser()

        input_vars = dict(content=content)
        input_vars.update({k: v for k, v in kwargs.items() if v})

        output = chain.invoke(input=input_vars, **kwargs)

        return dict(result=output)



if __name__ == "__main__":
    judger = Judger(langchain_model_name="gpt4o", standards_path="criteria_ordinary.jinja2", system_prompt_path="instructions.jinja2", process_path="process.jinja2", template_path="apply_rules.jinja2")
    output = judger(content="Hello, world!")
    print(output)