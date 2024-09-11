from typing import (
    Any,
)
from .toxicity import ToxicityModel, EvalRecord, Score
from promptflow.tracing import trace
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from buttermilk import BM
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser
from pathlib import Path
from typing import Literal
from buttermilk.utils.utils import read_text, read_yaml, scrub_serializable
from langchain_core.output_parsers import StrOutputParser

class Nemo(ToxicityModel):
    model: str
    process_chain: str = "langchain"
    standard: Literal["nemo_self_check.input", "nemo_self_check.output", "nemo_self_check.input_simple", "nemo_self_check.output_simple",]
    client: Any = None
    langchain_template: Any = None

    def init_client(self):
        bm = BM()
        llm = LLMs(connections=bm._connections_azure)[self.model]
        template_text = read_yaml(Path(__file__).parent / "templates/nemo_self_check.yaml")
        prompt_name = "nemo_self_check"
        criteria = str(self.standard).replace(prompt_name + ".", "")

        messages = [("system", template_text[prompt_name]["system"]), ("human", template_text[prompt_name][criteria])]
        langchain_template = ChatPromptTemplate.from_messages(messages, template_format="jinja2")

        chain = langchain_template | llm | ChatParser()
        return chain

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:

        input_vars = dict(content=text)
        input_vars.update({k: v for k, v in kwargs.items() if v})

        output = self.client.invoke(input=input_vars, **kwargs)

        return  output

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
                outcome.response = response

        except Exception as e:
            outcome.error = f"Unable to interpret result: {e}. {e.args}"
            outcome.response = response

        return outcome

class NemoInputSimpleGPT4o(Nemo):
    standard: str = "nemo_self_check.input_simple"
    model: str = "gpt4o"
class NemoInputComplexGPT4o(Nemo):
    standard: str = "nemo_self_check.input"
    model: str = "gpt4o"
class NemoOutputSimpleGPT4o(Nemo):
    standard: str = "nemo_self_check.output_simple"
    model: str = "gpt4o"
class NemoOutputComplexGPT4o(Nemo):
    standard: str = "nemo_self_check.output"
    model: str = "gpt4o"

class NemoInputSimpleLlama31_70b(Nemo):
    standard: str = "nemo_self_check.input_simple"
    model: str = "llama31_70b"
class NemoInputComplexLlama31_70b(Nemo):
    standard: str = "nemo_self_check.input"
    model: str = "llama31_70b"
class NemoOutputSimpleLlama31_70b(Nemo):
    standard: str = "nemo_self_check.output_simple"
    model: str = "llama31_70b"
class NemoOutputComplexLlama31_70b(Nemo):
    standard: str = "nemo_self_check.output"
    model: str = "llama31_70b"
