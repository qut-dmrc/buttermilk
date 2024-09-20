from typing import (
    Any,
)
from .toxicity import ToxicityModel, EvalRecord, Score, _Octo
from promptflow.tracing import trace
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from buttermilk import BM
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser
from pathlib import Path
from typing import Literal
from buttermilk.utils.utils import read_text, read_yaml, scrub_serializable
from langchain_core.output_parsers import StrOutputParser

import regex as re

class Nemo(ToxicityModel):
    standard: Literal["nemo_self_check.input", "nemo_self_check.output", "nemo_self_check.input_simple", "nemo_self_check.output_simple",]


    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord()

        try:
            answer = response.get('answer','').lower()
            if answer[:3] == 'yes':
                outcome.scores.append(
                        Score(measure=str(self.standard), score=1.0, result=True, reasons=[response.get('reasoning', str(response))])
                    )
                outcome.labels = [self.standard]
                outcome.predicted = True
            elif answer[:2] == "no":
                outcome.scores.append(
                            Score(measure=str(self.standard), score=0.0, result=False, reasons=[response.get('reasoning', str(response))])
                        )
                outcome.predicted = False
            else:
                outcome.error = f"Unable to interpret result."
                outcome.response = str(response)

        except Exception as e:
            raise ValueError(f"Unable to interpret result: {e}. {e.args}")

        return outcome
class NemoLangchain(Nemo):
    model: str
    client: Any = None
    process_chain: str = "langchain"

    def make_prompt(self, content: str) -> str:
        return content

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> str:

        input_vars = dict(content=prompt)
        input_vars.update({k: v for k, v in kwargs.items() if v})

        output = self.client.invoke(input=input_vars, **kwargs)

        return  output

    def init_client(self) -> None:
        bm = BM()
        llm = LLMs(connections=bm._connections_azure)[self.model]
        template_text = read_yaml(Path(__file__).parent / "templates/nemo_self_check.yaml")
        prompt_name = "nemo_self_check"

        criteria = re.match(pattern=r'^.*\.(.*)$', string=self.standard).group(1)

        messages = [("system", template_text[prompt_name]["system"]), ("human", template_text[prompt_name][criteria])]
        langchain_template = ChatPromptTemplate.from_messages(messages, template_format="jinja2")

        chain = langchain_template | llm | ChatParser()
        self.client = chain



class NemoInputSimpleMistralOcto(_Octo, Nemo):
    standard: str = "nemo_self_check.input_simple"
    model: str ="mistral-nemo-instruct"
class NemoInputComplexMistralOcto(Nemo, _Octo):
    standard: str = "nemo_self_check.input"
    model: str ="mistral-nemo-instruct"
class NemoOutputSimpleMistralOcto(Nemo, _Octo):
    standard: str = "nemo_self_check.output_simple"
    model: str ="mistral-nemo-instruct"
class NemoOutputComplexMistralOcto(Nemo, _Octo):
    standard: str = "nemo_self_check.output"
    model: str ="mistral-nemo-instruct"
class NemoInputSimpleGPT4o(NemoLangchain):
    standard: str = "nemo_self_check.input_simple"
    model: str = "gpt4o"
class NemoInputComplexGPT4o(NemoLangchain):
    standard: str = "nemo_self_check.input"
    model: str = "gpt4o"
class NemoOutputSimpleGPT4o(NemoLangchain):
    standard: str = "nemo_self_check.output_simple"
    model: str = "gpt4o"
class NemoOutputComplexGPT4o(NemoLangchain):
    standard: str = "nemo_self_check.output"
    model: str = "gpt4o"

class NemoInputSimpleLlama31_70b(NemoLangchain):
    standard: str = "nemo_self_check.input_simple"
    model: str = "llama31_70b"
class NemoInputComplexLlama31_70b(NemoLangchain):
    standard: str = "nemo_self_check.input"
    model: str = "llama31_70b"
class NemoOutputSimpleLlama31_70b(NemoLangchain):
    standard: str = "nemo_self_check.output_simple"
    model: str = "llama31_70b"
class NemoOutputComplexLlama31_70b(NemoLangchain):
    standard: str = "nemo_self_check.output"
    model: str = "llama31_70b"
