import os
from enum import Enum, EnumMeta
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Literal,
)

import regex as re
import torch
from huggingface_hub import login
from promptflow.tracing import trace
from pydantic import (
    Field,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from buttermilk.toxicity.toxicity import _HF, ToxicityModel
from buttermilk.utils.utils import read_yaml

from .types import EvalRecord, Score

TEMPLATE_DIR = Path(__file__).parent / "templates"


class LlamaGuardTemplate(Enum):
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


class MDJudge2Categories(Enum):
    O1 = "Toxic Content"
    O2 = "Unfair Representation"
    O3 = "Adult Content"
    O4 = "Erosion Of Trust in Public Information"
    O5 = "Propagating Misconceptions/False Beliefs"
    O6 = "Risky Financial Practices"
    O7 = "Trade and Compliance"
    O8 = "Dissemination Of Dangerous Information"
    O9 = "Privacy Infringement"
    O10 = "Security Threats"
    O11 = "Defamation"
    O12 = "FraudOr Deceptive Action"
    O13 = "Influence Operations"
    O14 = "Illegal Activities"
    O15 = "Persuasion and Manipulation"
    O16 = "Violation Of Personal Property"


class LlamaGuardTox(ToxicityModel):
    categories: EnumMeta
    template: str
    client: Any = None
    tokenizer: Any = None
    model: str
    options: ClassVar[dict] = dict()

    @trace
    def make_prompt(self, content):
        # Load the message info into the output
        agent_type = "Agent"
        content = "[INST] " + self.template.format(prompt=content, agent_type=agent_type) + "[/INST]"

        return content

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        reasons = []
        explanation = ""
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
            for r in labels.split(","):
                if r != "":
                    r = str.upper(r)
                    reasons.append(f"{r}: {self.categories[r].value}")
        except KeyError:
            pass  # could not find lookup string, just use the value we received
        except (IndexError, ValueError):
            pass  # Unknown reason

        # Load the message info into the output
        outcome = EvalRecord()

        if answer[:4] == "safe":
            outcome.prediction = False
            outcome.labels = ["safe"] + reasons
            explanation = explanation or "safe"
            for reason in reasons:
                outcome.scores.append(
                    Score(
                        measure=str(reason).upper(),
                        reasons=[explanation],
                        score=0.0,
                        result=False,
                    ),
                )

        elif answer[:6] == "unsafe":
            outcome.prediction = True
            explanation = explanation or "unsafe"
            if not reasons:
                outcome.error = f"Invalid reasons returned from LLM: {answer}"

            for reason in reasons:
                outcome.scores.append(
                    Score(
                        measure=str(reason).upper(),
                        reasons=[explanation],
                        score=1.0,
                        result=True,
                    ),
                )
            outcome.labels = ["unsafe"] + reasons

        else:
            raise ValueError(f"Unexpected response from LLM: {response}")

        return outcome


class LlamaGuard1Together(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories1
    template: str = Field(
        default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD1),
    )
    standard: str = "llamaguard1"
    model: str = "Meta-Llama/Llama-Guard-7b"
    process_chain: str = "together"
    options: ClassVar[dict] = dict(temperature=1.0, top_k=1)
    client: "Together" = None

    def init_client(self) -> None:
        self.client = Together(model=self.model, **self.options)


class LlamaGuard1Replicate(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories1
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD1))
    model: str = "tomasmcm/llamaguard-7b:86a2d8b79335b1557fc5709d237113aa34e3ae391ee46a68cc8440180151903d"
    standard: str = "llamaguard1"
    process_chain: str = "replicate"
    client: Any = None

    def init_client(self) -> None:
        self.client = Replicate(model=self.model, **self.options)


class LlamaGuard2Replicate(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    model: str = "meta/meta-llama-guard-2-8b:b063023ee937f28e922982abdbf97b041ffe34ad3b35a53d33e1d74bb19b36c4"
    standard: str = "llamaguard2"
    process_chain: str = "replicate"
    client: Any = None

    def init_client(self) -> None:
        self.client = Replicate(model=self.model, **self.options)


class LlamaGuard2Together(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    standard: str = "llamaguard2"
    model: str = "meta-llama/LlamaGuard-2-8b"
    process_chain: str = "together"
    client: "Together" = None
    options: ClassVar[dict] = dict(temperature=1.0)

    def init_client(self) -> None:
        self.client = Together(model=self.model, **self.options)


class LlamaGuard2Local(_HF, LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    standard: str = "llamaguard2"
    process_chain: str = "local transformers"
    model: str = "meta-llama/Meta-Llama-Guard-2-8B"
    client: Any = None
    options: ClassVar[dict] = dict(temperature=1.0)
    call_options: ClassVar[dict] = dict(max_new_tokens=128)


class LlamaGuard2HF(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    model: str = "meta-llama/Meta-Llama-Guard-2-8B"
    standard: str = "llamaguard2"
    process_chain: str = "huggingface API"
    client: Any = None

    def init_client(self) -> None:
        from buttermilk.libs import HFInferenceClient

        self.client = HFInferenceClient(hf_model_path=self.model, **self.options)


class _LlamaGuard3Common(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(
        default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3),
    )
    standard: str = "llamaguard3"

    def make_prompt(self, content):
        agent_type = "Agent"
        content = f"{agent_type}: {content}"
        content = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|> "
            + self.template.format(prompt=content, agent_type=agent_type)
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        return content


class LlamaGuard3Local(_HF, _LlamaGuard3Common):
    model: str = "meta-llama/Llama-Guard-3-8B"
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    standard: str = "llamaguard3"
    process_chain: str = "local transformers"
    options: ClassVar[dict] = dict(temperature=1.0)
    call_options: ClassVar[dict] = dict(max_new_tokens=128, pad_token_id=0)


class LlamaGuard3LocalInt8(LlamaGuard3Local):
    model: str = "meta-llama/Llama-Guard-3-8B-INT8"
    device: str = "cuda"

    def init_client(self) -> None:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = 0
        self.client = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            quantization_config=quantization_config,
            **self.options,
        )


class LlamaGuard3Together(_LlamaGuard3Common):
    model: str = "meta-llama/Meta-Llama-Guard-3-8B"
    process_chain: str = "Together API"

    def init_client(self) -> None:
        self.client = Together(model=self.model, **self.options)


# MDJudge has the same response style as LlamaGuard
class MDJudgeLocal(LlamaGuardTox):
    process_chain: str = "local transformers"
    model: str = "OpenSafetyLab/MD-Judge-v0.1"
    client: Any = None
    tokenizer: Any = None
    template: str

    def init_client(self) -> None:
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.client = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def make_prompt(self, content):
        prompt = "User: go on...\nAgent: " + content

        prompt = "[INST] " + self.template.format(prompt=prompt) + " [/INST]"

        return content


class MDJudgeLocalDomain(MDJudgeLocal):
    standard: Literal["MDJUDGE.DOMAIN_POLICY"] = "MDJUDGE.DOMAIN_POLICY"
    template: str = Field(
        default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGEDOMAIN),
    )
    categories: EnumMeta = MDJudgeDomainCategories


class MDJudgeLocalTask(MDJudgeLocal):
    standard: Literal["MDJUDGE.TASK_POLICY"] = "MDJUDGE.TASK_POLICY"
    template: str = Field(
        default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGETASK),
    )
    categories: EnumMeta = MDJudgeTaskCategories


class MDJudge2(MDJudgeLocal):
    process_chain: str = "local transformers"
    model: str = "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b"
    standard: str = "MDJudge_v0.2"
    client: Any = None
    tokenizer: Any = None
    template: str = Field(
        default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGE2),
    )
    categories: EnumMeta = MDJudge2Categories

    def init_client(self) -> None:
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
        )
        self.client = AutoModelForCausalLM.from_pretrained(
            self.model,
            trust_remote_code=True,
        ).to("cuda")

    def make_prompt(self, content: str) -> str:
        input_conversation = [
            {"role": "user", "content": self.template.strip().format(prompt=content)},
        ]
        prompt = self.tokenizer.apply_chat_template(input_conversation, tokenize=False)

        return prompt

    @trace
    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")

        outputs = self.client.generate(**inputs, max_new_tokens=256)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        resp = self.tokenizer.decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        resp = resp.split("<|im_end|>")[0]
        return resp.strip()

    # sample response:
    #         response = """The agent message provided is harmful. It contains derogatory language ('SISSY') which is offensive and can be considered a slur. Additionally, it directs the user to a specific tweet, which could potentially lead to further harmful content or interactions. The use of such language and the promotion of potentially offensive content can contribute to a hostile environment and emotional harm in the conversation.
    # [RESULT]: unsafe, score 4
    # [UNSAFE CATEGORY]: O1: Toxic Content"""
    @trace
    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord()
        reasons = []

        try:
            reasons, result, category = response.strip().split("\n")
            if "unsafe" in result.lower():
                outcome.prediction = True
                outcome.labels.append("unsafe")
            elif "safe" in result.lower():
                outcome.prediction = False

            score = re.match(r".*score (\d+)", result).group(1)
            if category := re.match(r".*UNSAFE CATEGORY]: (.*)", category):
                category = category.group(1)
                outcome.labels.append(category)
                outcome.scores.append(
                    Score(measure=category, severity=float(score), reasons=[reasons]),
                )

        except Exception as e:
            outcome.error = f"Unable to interpret result: {e}. {e.args}"
            outcome.response = str(response)

        return outcome
