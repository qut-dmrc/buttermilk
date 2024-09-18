
import os
from enum import Enum, EnumMeta, StrEnum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Literal,
    Union,
)

from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain_community.llms import Replicate
from langchain_together import Together
from promptflow.tracing import trace
from pydantic import (
    Field,
)
from buttermilk.apis import HFInferenceClient
from buttermilk.toxicity.toxicity import _Octo, ToxicityModel
from buttermilk.utils.utils import read_yaml
from ..types.tox import EvalRecord, Score
TEMPLATE_DIR = Path(__file__).parent / 'templates'

class LlamaGuardTemplate(StrEnum):
    LLAMAGUARD1 = "llamaguard1"
    LLAMAGUARD2 = "llamaguard2"
    LLAMAGUARD3 = "llamaguard3"
    MDJUDGEDOMAIN = "mdjudgedomain"
    MDJUDGETASK = "mdjudgetask"
    MDJUDGE2 = "mdjudge2"

def llamaguard_template(template: LlamaGuardTemplate):
    tpl = read_yaml(Path(__file__).parent / "templates/llamaguard.yaml")
    return tpl[template.value]

class LlamaGuardUnsafeContentCategories1(Enum):
    O1 = "Violence & Hate"
    O2 = "Sexual Content"
    O3 = "Criminal Planning"
    O4 = "Guns & Illegal Weapons"
    O5 = "Regulated or Controlled Substances "
    O6 = "Suicide & Self-Harm"

class LlamaGuardUnsafeContentCategories2(Enum):
    S1 = "Violent Crimes"
    S2 = "Non-Violent Crimes"
    S3 = "Sex Crimes"
    S4 = "Child Exploitation"
    S5 = "Specialized Advice"
    S6 = "Privacy"
    S7 = "Intellectual Property"
    S8 = "Indiscriminate Weapons"
    S9 = "Hate"
    S10 = "Self-Harm"
    S11 = "Sexual Content"

class LlamaGuardUnsafeContentCategories3(Enum):
    S1 = "Violent Crimes"
    S2 = "Non-Violent Crimes"
    S3 = "Sex-Related Crimes"
    S4 = "Child Sexual Exploitation"
    S5 = "Defamation"
    S6 = "Specialized Advice"
    S7 = "Privacy"
    S8 = "Intellectual Property"
    S9 = "Indiscriminate Weapons"
    S10 = "Hate"
    S11 = "Suicide & Self-Harm"
    S12 = "Sexual Content"
    S13 = "Elections"
    S14 = "Code Interpreter Abuse"

class MDJudgeDomainCategories(Enum):
    O1 = "Representation & Toxicity"
    O2 = "Misinformation Harms"
    O3 = "Socioeconomic Harms"
    O4 = "Information & Safety"
    O5 = "Malicious Use"
    O6 = "Human Autonomy & Integrity"

class MDJudgeTaskCategories(Enum):
    O1 = "Toxic Content"
    O2 = "Unfair Representation"
    O3 = "Adult Content"
    O4 = "Erosion of Trust in Public Information"
    O5 = "Propagating Misconceptions/False Beliefs"
    O6 = "Risky Financial Practices"
    O7 = "Trade and Compliance"
    O8 = "Dissemination of Dangerous Information"
    O9 = "Privacy Infringement"
    O10 = "Security Threats"
    O11 = "Defamation"
    O12 = "Fraud or Deceptive Action"
    O13 = "Influence Operations"
    O14 = "Illegal Activities"
    O15 = "Persuasion and Manipulation"
    O16 = "Violation of Personal Property"

class LlamaGuardTox(ToxicityModel):
    categories: EnumMeta
    template: str
    client: Any = None
    tokenizer: Any = None
    model: str
    options: ClassVar[dict] = dict(temperature=1.0, max_new_tokens=128, top_k=1)

    def make_prompt(self, content):
        # Load the message info into the output
        agent_type = "Agent"
        content = (
            "[INST] "
            + self.template.format(prompt=content, agent_type=agent_type)
            + "[/INST]"
        )

        return content

    @trace
    def call_client(
        self, content: str, **kwargs
    ) -> Any:
        prompt = self.make_prompt(content)
        return self.client(prompt, **kwargs)

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        reasons = []
        explanation = ''
        try:
            answer, labels, explanation = response.strip().split("\n")
        except ValueError:
            try:
                answer, labels = response.strip().split(" ", 3)
            except ValueError:
                try:
                    answer, labels = response.strip().split("\n")
                except ValueError:
                    try:
                        answer, labels = response.strip().split(" ")
                    except ValueError:
                        try:
                            answer = response.strip()
                        except ValueError:
                            answer = response
                        labels = ""


        try:
            for r in labels.split(','):
                if r != '':
                    r = str.upper(r)
                    reasons.append(f"{r}: {self.categories[r].value}")
        except KeyError:
            pass  # could not find lookup string, just use the value we received
        except (IndexError, ValueError):
            pass  # Unknown reason

        # Load the message info into the output
        outcome = EvalRecord(
        )

        if answer[:4] == "safe":
            outcome.predicted = False
            outcome.labels = ["safe"] + reasons
            explanation = explanation or 'safe'
            for reason in reasons:
                outcome.scores.append(
                    Score(measure=str(reason).upper(),
                        reasons=[explanation], score=0.0, result=False
                    )
                )

        elif answer[:6] == "unsafe":
            outcome.predicted = True
            explanation = explanation or 'unsafe'
            if not reasons:
                reasons = ["unknown"]
                outcome.error = f"Invalid reasons returned from LLM: {answer}"

            for reason in reasons:
                outcome.scores.append(
                    Score(measure=str(reason).upper(),
                        reasons=[explanation], score=1.0, result=True,
                    )
                )
            outcome.labels = ["unsafe"] + reasons

        else:
            raise ValueError(f"Unexpected response from LLM: {response}")

        return outcome

class LlamaGuardToxLocal(LlamaGuardTox):
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA)",
    )
    options: ClassVar[dict] = dict(temperature=1.0,pad_token_id=0, max_new_tokens=128, top_k=1)

    def init_client(self):
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.client = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype=torch.bfloat16)
        return self.client

    @trace
    def call_client(
        self, content: str, **kwargs
    ) -> Any:
        prompt = self.make_prompt(content)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)['input_ids']
        output = self.client.generate(input_ids=input_ids, **self.options, **kwargs)
        prompt_len = input_ids.shape[-1]
        response = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        try:
            result = response[0][0]['generated_text'].strip()
            return str(result[len(prompt):])
        except:
            try:
                result = response.generations[0][0].text.strip()
                return str(result[len(prompt):])
            except:
                result = response.strip()
                return result

class LlamaGuard1Together(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories1
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD1))
    standard: str = "llamaguard1"
    model: str = "Meta-Llama/Llama-Guard-7b"
    process_chain: str = "together"
    options: ClassVar[dict] = dict(temperature=1.0, top_k=1, top_p=0.95)
    client: Together = None

    def init_client(self):
        return Together(model=self.model, **self.options)

class LlamaGuard1Replicate(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories1
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD1))
    model: str = "tomasmcm/llamaguard-7b:86a2d8b79335b1557fc5709d237113aa34e3ae391ee46a68cc8440180151903d"
    standard: str = "llamaguard1"
    process_chain: str = "replicate"
    client: Any = None

    def init_client(self):
        return Replicate(model=self.model, **self.options)


class LlamaGuard2Replicate(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    model: str = "meta/meta-llama-guard-2-8b:b063023ee937f28e922982abdbf97b041ffe34ad3b35a53d33e1d74bb19b36c4"
    standard: str = "llamaguard2"
    process_chain: str = "replicate"
    client: Any = None

    def init_client(self):
            return Replicate(model=self.model, **self.options)


class LlamaGuard2Together(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    standard: str = "llamaguard2"
    model: str = "meta-llama/LlamaGuard-2-8b"
    process_chain: str = "together"
    client: Together = None
    options: ClassVar[dict] = dict(temperature=1.0, top_p=0.95)

    def init_client(self):
        return Together(model=self.model, **self.options)


class LlamaGuard2Local(LlamaGuardToxLocal):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    standard: str = "llamaguard2"
    process_chain: str = "local transformers"
    model: str = "meta-llama/Meta-Llama-Guard-2-8B"
    client: Any = None


class LlamaGuard2HF(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    model: str = "meta-llama/Meta-Llama-Guard-2-8B"
    standard: str = "llamaguard2"
    process_chain: str = "huggingface API"
    client: Any = None

    def init_client(self):
        return HFInferenceClient(hf_model_path=self.model, **self.options)



class _LlamaGuard3Common(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    standard: str = "llamaguard3"
    options: ClassVar[dict] = dict(temperature=1.0)

    def make_prompt(self, content):
        agent_type = "Agent"
        content = f"{agent_type}: {content}"
        content = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|> " +
            self.template.format(prompt=content, agent_type=agent_type) +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>")

        return content

class LlamaGuard3Local(LlamaGuardToxLocal):
    model: str = "meta-llama/Llama-Guard-3-8B"
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    standard: str = "llamaguard3"
    process_chain: str = "local transformers"

    def make_prompt(self, content):
        agent_type = "Agent"
        content = f"{agent_type}: {content}"
        content = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|> " +
            self.template.format(prompt=content, agent_type=agent_type) +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>")

        return content


class LlamaGuard3LocalInt8(LlamaGuard3Local):
    model: str = "meta-llama/Llama-Guard-3-8B-INT8"
    device: str  = "cuda"
    dtype: Any = "auto"
    options: ClassVar[dict] = dict(temperature=1.0)

    def init_client(self):
        quantization_config  = BitsAndBytesConfig(load_in_8bit=True)
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.client = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=self.dtype, device_map=self.device, quantization_config=quantization_config, **self.options)
        return self.client
class LlamaGuard3Together(_LlamaGuard3Common):
    model: str = "meta-llama/Meta-Llama-Guard-3-8B"
    process_chain: str = "Together API"
    options: ClassVar[dict] = dict(temperature=1.0)

    def init_client(self):
        return Together(model=self.model, **self.options)

class LlamaGuard3Octo(_Octo, _LlamaGuard3Common):
    model: str = "llama-guard-3-8b"
    process_chain: str = "Octo API"


## MDJudge has the same response style as LlamaGuard
class MDJudgeLocal(LlamaGuardTox):
    process_chain: str = "local transformers"
    model: str = "OpenSafetyLab/MD-Judge-v0.1"
    client: Any = None
    tokenizer: Any = None
    template: str

    def init_client(self):
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.client = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype=torch.bfloat16)
        return self.client

    def make_prompt(self, content):
        prompt = "User: go on...\nAgent: " + content

        prompt = "[INST] " + self.template.format(prompt=prompt) + " [/INST]"

        return content



class MDJudgeLocalDomain(MDJudgeLocal):
    standard: Literal["MDJUDGE.DOMAIN_POLICY"] = "MDJUDGE.DOMAIN_POLICY"
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGEDOMAIN))
    categories: EnumMeta = MDJudgeDomainCategories


class MDJudgeLocalTask(MDJudgeLocal):
    standard: Literal["MDJUDGE.TASK_POLICY"] = "MDJUDGE.TASK_POLICY"
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGETASK))
    categories: EnumMeta = MDJudgeTaskCategories

class MDJudge2(MDJudgeLocal):
    process_chain: str = "local transformers"
    model: str = "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b"
    standard: str = 'MDJudge_v0.2'
    client: Any = None
    tokenizer: Any = None
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGE2))

    def init_client(self):
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.client = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True).to("cuda")
        return self.client


    def make_prompt(self, content):
        input_conversation = [
        {"role": "user", "content": self.template.strip().format(prompt=content)}
        ]

        return input_conversation

    @trace
    def call_client(
        self, content: str, **kwargs
    ) -> Any:
        prompt = self.make_prompt(content)
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

        outputs = self.client.generate(**inputs, max_new_tokens=256)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        resp = self.tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        resp = resp.split("<|im_end|>")[0]
        return resp.strip()
