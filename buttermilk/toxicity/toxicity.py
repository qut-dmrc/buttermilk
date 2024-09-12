from __future__ import annotations

import abc
import os
from enum import Enum, EnumMeta, IntEnum, StrEnum
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    ClassVar,
    List,
    Literal,
    LiteralString,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import boto3
from buttermilk.exceptions import RateLimit
from buttermilk.runner._runner_types import Job
import evaluate
import google.auth
import numpy as np
import openai
import pandas as pd
import psutil
import requests
import torch
import transformers
import urllib3
from transformers import AutoTokenizer, AutoModelForCausalLM
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import RateLimitError as AnthropicRateLimitError
from azure.ai.contentsafety import BlocklistClient, ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions,
    AnalyzeTextOutputType,
    TextCategory,
)
from azure.cognitiveservices.vision.contentmoderator import ContentModeratorClient
from azure.cognitiveservices.vision.contentmoderator.models import Screen
from azure.core.credentials import AzureKeyCredential
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types.generation_types import (
    BlockedPromptException,
    StopCandidateException,
)
from transformers import pipeline
from googleapiclient import discovery
from langchain_community.llms import Replicate
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_together import Together
from msrest.authentication import CognitiveServicesCredentials
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import RateLimitError as OpenAIRateLimitError
from promptflow.tracing import trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
    root_validator,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_random_exponential,
)

from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.utils import read_text, read_yaml, scrub_serializable
from buttermilk.apis import HFTransformer, HFInferenceClient

TEMPLATE_DIR = Path(__file__).parent / 'templates'

PerspectiveAttributes = Literal[
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
    "SEXUALLY_EXPLICIT",
]
PerspectiveAttributesExperimental = Literal[
    "TOXICITY_EXPERIMENTAL",
    "SEVERE_TOXICITY_EXPERIMENTAL",
    "IDENTITY_ATTACK_EXPERIMENTAL",
    "INSULT_EXPERIMENTAL",
    "PROFANITY_EXPERIMENTAL",
    "THREAT_EXPERIMENTAL",
    "SEXUALLY_EXPLICIT",
    "FLIRTATION",
    ## These are 'bridging' attributes: https://medium.com/jigsaw/announcing-experimental-bridging-attributes-in-perspective-api-578a9d59ac37
    "AFFINITY_EXPERIMENTAL",
    "COMPASSION_EXPERIMENTAL",
    "CURIOSITY_EXPERIMENTAL",
    "NUANCE_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "REASONING_EXPERIMENTAL",
    "RESPECT_EXPERIMENTAL",
]

class LlamaGuardTemplate(StrEnum):
    LLAMAGUARD1 = "llamaguard1"
    LLAMAGUARD2 = "llamaguard2"
    LLAMAGUARD3 = "llamaguard3"
    MDJUDGEDOMAIN = "mdjudgedomain"
    MDJUDGETASK = "mdjudgetask"

def llamaguard_template(template: LlamaGuardTemplate):
    tpl = read_yaml(Path(__file__).parent / "templates/llamaguard.yaml")
    return tpl[template.value]

from ..types.tox import EvalRecord, Score

# Let's provide an interface for all the various toxicity models
class ToxicityModel(BaseModel):
    model: str
    process_chain: str
    standard: str
    client: Any = None
    info_url: Optional[str] = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_model(self):
        if self.client is None:
            self.client = self.init_client()
            if self.client is None:
                raise ValueError(f"Unable to initialize client for {self.model}")
        return self

    def init_client(self):
        if self.client is None:
            raise NotImplementedError()
        return self.client

    def run(self, job: Job) -> Job:
        response = self.moderate(job.record.text)
        if not isinstance(response, EvalRecord):
            raise ValueError(f"Expected an EvalRecord from toxicity model, got: {type(response)} for: {job}")

        # add identifying info in
        response.model=self.model
        response.process=self.process_chain
        response.standard=self.standard
        response.record_id=job.record.record_id

        output = job.model_copy()
        output.result = response
        return output

    # @retry(
    #     retry=retry_if_exception_type(
    #         exception_types=(
    #             RateLimit,
    #             requests.exceptions.ConnectionError,
    #             urllib3.exceptions.ProtocolError,urllib3.exceptions.TimeoutError,
    #             OpenAIAPIConnectionError, OpenAIRateLimitError, AnthropicAPIConnectionError, AnthropicRateLimitError, ResourceExhausted
    #         ),
    #     ),
    #         # Wait interval: increasing exponentially up to a max of 30s between retries
    #         wait=wait_exponential_jitter(initial=1, max=30, jitter=5),
    #         # Retry up to five times before giving up
    #         stop=stop_after_attempt(5),
    # )
    @trace
    def __call__(self, content: str, record_id: Optional[str] = None, **kwargs) -> dict:
        if not content or content.strip() == '':
            if (content := kwargs.get('text','').strip()) == '':
                response = EvalRecord(
                    error="No input provided."
                    )
        else:
            try:
                response = self.moderate(content)
                if not isinstance(response, EvalRecord):
                    response = EvalRecord( **response)

            except Exception as e:
                response = EvalRecord(
                    error=f'{e}',
                    metadata=extract_error_info(e=e)
                )

        # add identifying info in
        response.model=self.model
        response.process=self.process_chain
        response.standard=self.standard
        response.record_id=record_id

        output = response.model_dump()
        output = scrub_serializable(output)
        return output

    async def moderate_async(
            self, text: str
        ) -> EvalRecord:
            return self.moderate(text)

    @trace
    def moderate(
        self, text: str, **kwargs
    ) -> EvalRecord:
        response = self.call_client(text, **kwargs)
        output = self.interpret(response)
        return output

    @trace
    @abc.abstractmethod
    def call_client(
        self, text: str, *args, **kwargs
    ) -> Any:
        raise NotImplementedError()

    @trace
    @abc.abstractmethod
    def interpret(self, response: Any) -> EvalRecord:
        raise NotImplementedError()


class Perspective(ToxicityModel):
    model: str = "perspective"
    process_chain: str = "api"
    standard: str = "perspective"
    client: Any = None

    def init_client(self):
        credentials, _ = google.auth.default()
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            credentials=credentials,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        return client


    @trace
    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(predicted=False)
        for key, value in response["attributeScores"].items():
            score = Score(measure=key, score=float(value["summaryScore"]["value"]))
            outcome.scores.append(score)
            if score.score and score.score > 0.5:
                outcome.labels.append(key)
                if key.lower() in ["toxicity", "severe_toxicity","toxicity_experimental", "severe_toxicity_experimental"]:
                    outcome.predicted = True

        return outcome

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        if not (attributes := kwargs.get('attributes')):
            # get all
            attributes = (
                PerspectiveAttributes.__args__
                + PerspectiveAttributesExperimental.__args__
            )

        analyze_request = {
            "comment": {"text": text},
            "requestedAttributes": {attr: {} for attr in attributes},
            "doNotStore": True,
        }

        response = self.client.comments().analyze(body=analyze_request).execute()

        return response


class Comprehend(ToxicityModel):
    model: str = "comprehend"
    process_chain: str = "api"
    standard: str = "comprehend"
    client: Any = None

    def init_client(self):
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION")

        return boto3.client(
            service_name="comprehend",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        return self.client.detect_toxic_content(
            LanguageCode="en", TextSegments=[{"Text": text}]
        )

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(
            predicted=False,
        )
        try:
            result = response["ResultList"][0]
        except Exception as e:
            raise ValueError(f"Unable to interpret response from AWS Comprehend. Response: {response}. Error: {e} {e.args=}"
            )

        outcome.scores = [Score(measure="Toxicity", score=result["Toxicity"])]
        if result["Toxicity"] > 0.5:
            outcome.labels.append("Toxicity")
            outcome.predicted = True
        else:
            outcome.predicted = False

        for score in result["Labels"]:
            new_score = Score(measure=score["Name"], score=score["Score"])
            outcome.scores.append(new_score)
            if new_score.score and new_score.score > 0.5:
                outcome.labels.append(score["Name"])

        return outcome


class AzureContentSafety(ToxicityModel):
    model: str = "AzureContentSafety"
    process_chain: str = "api"
    standard: str = "AzureContentSafety 2023-10-01"
    client: ContentSafetyClient = None

    """ Azure Content Safety API: https://aka.ms/acs-doc
        https://contentsafety.cognitive.azure.com/
        Categories:
            Hate 	Hate and fairness-related harms refer to any content that attacks or uses pejorative or discriminatory language with reference to a person or identity group based on certain differentiating attributes of these groups including but not limited to race, ethnicity, nationality, gender identity and expression, sexual orientation, religion, immigration status, ability status, personal appearance, and body size.
            Sexual 	Sexual describes language related to anatomical organs and genitals, romantic relationships, acts portrayed in erotic or affectionate terms, pregnancy, physical sexual acts, including those portrayed as an assault or a forced sexual violent act against one's will, prostitution, pornography, and abuse.
            Violence 	Violence describes language related to physical actions intended to hurt, injure, damage, or kill someone or something; describes weapons, guns and related entities, such as manufacturers, associations, legislation, and so on.
            Self-harm 	Self-harm describes language related to physical actions intended to purposely hurt, injure, damage one's body or kill oneself.

        Severity levels (Text):
            The current version of the text model supports the full 0-7 severity scale. By default, the response will output 4 values: 0, 2, 4, and 6. Each two adjacent levels are mapped to a single level. Users could use "outputType" in request and set it as "EightSeverityLevels" to get 8 values in output: 0,1,2,3,4,5,6,7.

        **Note: We are using a cut-off of >=3 for the binary 'result' field.**
    """

    def init_client(self):
        API_KEY = os.environ["AZURE_CONTENT_SAFETY_KEY"]
        ENDPOINT = os.environ.get(
            "AZURE_CONTENT_SAFETY_ENDPOINT",
            "https://westus.api.cognitive.microsoft.com",
        )

        credential = AzureKeyCredential(API_KEY)
        content_safety_client = ContentSafetyClient(ENDPOINT, credential)
        # blocklist_client = BlocklistClient(endpoint, credential)
        return content_safety_client

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        request = AnalyzeTextOptions(
            text=text, output_type=AnalyzeTextOutputType.EIGHT_SEVERITY_LEVELS
        )
        return self.client.analyze_text(request)

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        # Load the message info into the output
        outcome = EvalRecord(
            predicted=False,
        )

        for item in response.categories_analysis:
            measure = None
            score_labels = []
            severity_score = None
            try:
                measure = item.category
                severity = item.severity
                score_labels = [f"Severity: {severity}"]
                # arbitrary cutoff here. better to use the Score() value.
                if severity >= 3:
                    outcome.predicted = True
                    outcome.labels.append(measure)
                    score_labels.append(measure)

                severity_score = severity
            except Exception as e:
                raise ValueError(f"Unable to interpret Azure content safety score: {item}.")

            if measure is not None:
                outcome.scores.append(
                    Score(
                        measure=measure,
                        score=severity_score,
                        labels=score_labels,
                    )
                )

        return outcome


class AzureModerator(ToxicityModel):
    model: str = "azure content-moderator"
    process_chain: str = "text-moderation-api"
    standard: str = "Azure Content Moderator"
    client: ContentModeratorClient = None

    """ Azure Content Moderator screen text
        https://learn.microsoft.com/en-us/azure/ai-services/content-moderator/overview

        **Azure Content Moderator is deprecated as of February 2024 and will be retired by February 2027. It is replaced by Azure AI Content Safety, which offers advanced AI features and enhanced performance.**

        Terms: If the API detects any profane terms in any of the supported languages, those terms are included in the response. Profanity detection uses term-based matching with built-in list of profane terms in various languages

        Classification:
            Category1 refers to potential presence of language that may be considered sexually explicit or adult in certain situations.
            Category2 refers to potential presence of language that may be considered sexually suggestive or mature in certain situations.
            Category3 refers to potential presence of language that may be considered offensive in certain situations.
            Score is between 0 and 1. The higher the score, the higher the model is predicting that the category may be applicable. This feature relies on a statistical model rather than manually coded outcomes. We recommend testing with your own content to determine how each category aligns to your requirements.
            ReviewRecommended is either true or false depending on the internal score thresholds. Customers should assess whether to use this value or decide on custom thresholds based on their content policies.
    """


    def init_client(self):
        SUBSCRIPTION_KEY = os.environ["AZURE_CONTENT_MODERATOR_KEY"]
        ENDPOINT = os.environ.get(
            "AZURE_CONTENT_MODERATOR_ENDPOINT",
            "https://westus.api.cognitive.microsoft.com",
        )
        return ContentModeratorClient(
            endpoint=ENDPOINT,
            credentials=CognitiveServicesCredentials(subscription_key=SUBSCRIPTION_KEY),
        )

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        text_stream = StringIO(text)
        response = self.client.text_moderation.screen_text(
            language="eng",
            text_content_type="text/plain",
            text_content=text_stream,
            autocorrect=True,
            pii=True,
            classify=True,
        ).as_dict()

        return response

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        # Load the message info into the output
        outcome = EvalRecord(
            predicted=False,
        )
        categories = {
            "category1": "sexually explicit or adult",
            "category2": "sexually suggestive or mature",
            "category3": "offensive",
        }

        for key, label in categories.items():
            score = Score(measure=label, score=response["classification"][key]["score"])

            outcome.scores.append(score)

            if score.score and score.score >= 0.5:
                outcome.labels.append(label)

        outcome.predicted = response["classification"]["review_recommended"]

        # Extra info usually includes  ["status", "language", r"tracking_id", "normalized_text", "auto_corrected_text"]
        # discard the ones we don't need.
        _result_keys = [
            "original_text",
            "classification",
            "normalized_text",
            "auto_corrected_text"
        ]
        outcome.metadata = {
            x: response[x] for x in response.keys() if x not in _result_keys
        }

        return outcome


class REGARD(ToxicityModel):
    model: str = "regard"
    process_chain: str = "evaluate"
    standard: str = "regard"
    client: Any = None

    def init_client(self):
        if torch.cuda.is_available():
            return evaluate.load("regard", module_type="measurement",device = "cuda")
        else:
            return evaluate.load("regard", module_type="measurement")

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        result = self.client.compute(data=[text])
        return result

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        result = response["regard"][0]

        # Load the message info into the output
        outcome = EvalRecord(
            predicted=False,
        )
        for record in result:
            score = Score(measure=record["label"], confidence=record["score"])
            outcome.scores.append(score)
            if record["score"] >= 0.5:
                outcome.labels.append(score.measure)
                if score.measure == "negative":
                    outcome.predicted = True

        return outcome


class HONEST(ToxicityModel):
    model: str = "honest"
    process_chain: str = "evaluate"
    standard: str = "honest"
    client: Any = None

    def init_client(self):
        return evaluate.load("honest", "en")
    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        completions = [text.split(" ")]
        result = self.client.compute(predictions=completions)
        return result

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(
        )
        outcome.scores = [Score(measure="honest", score=response["honest_score"])]
        outcome.predicted = response["honest_score"] > 0.5
        if outcome.predicted:
            pass  # TODO: add label if we have it

        return outcome


class LFTW(ToxicityModel):
    model: str = "lftw_r4"
    process_chain: str = "hf_transformers"
    standard: str = "lftw_r4-target"
    client: Any = None

    def init_client(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return transformers.pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
                device=device
        )
    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        return self.client(text)

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        if len(response) > 1:
            raise ValueError("Expected only one result from LFTW model")
        result = response[0]

        # Load the message info into the output
        outcome = EvalRecord(
        )
        outcome.scores = [Score(measure=result["label"], confidence=result["score"])]
        outcome.predicted = result["label"] == "hate"
        outcome.labels = [result["label"]]

        return outcome


class GPTJT(ToxicityModel):
    # Load model directly
    model: str = "togethercomputer/GPT-JT-Moderation-6B"
    process_chain: str = "hf_transformers"
    standard: str = "gpt-jt-mod-v1"
    template: str = Field(default_factory=lambda: read_text(TEMPLATE_DIR / "gpt-jt-mod-v1.txt"))
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA)",
    )

    client: Any = None

    ResponseMap: dict[str, int] = {
        "casual": 1,
        "needs caution": 2,
        "needs intervention": 3,
        "possibly needs caution": 4,
        "probably needs caution": 5,
    }

    def init_client(self):
        # Use a pipeline as a high-level helper
        return pipeline(
            "text-generation", model="togethercomputer/GPT-JT-Moderation-6B",
                device=self.device,max_new_tokens=3
        )
    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        prompt = self.template.format(content=text)

        response = self.client(prompt)

        if len(response) > 1:
            raise ValueError("Expected only one result from model")
        result = response[0]['generated_text']
        if result.startswith(prompt):
            result = result[len(prompt):]

        return result

    @trace
    def interpret(self, response: Any) -> EvalRecord:

        # Load the message info into the output
        outcome = EvalRecord(
        )
        outcome.response = response
        try:
            outcome.scores = [
                Score(
                    measure=self.standard, score=self.ResponseMap[response], labels=[response]
                )
            ]

            outcome.predicted = self.ResponseMap[response] >= 2
            outcome.labels = [response]
        except (TypeError, ValueError) as e:
            outcome.error = f"Unable to extract scores. Hit error: {e} {e.args}"

        return outcome


###
# todo: add https://github.com/unitaryai/detoxify
####


class OpenAIModerator(ToxicityModel):
    model: str = "openaimod"
    process_chain: str = "api"
    standard: str = "openaimod"
    client: Any = None

    def init_client(self):
        openai.api_type = 'openai'
        return openai.moderations

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        return self.client.create(input=text)

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        if len(response.results) > 1:
            raise ValueError("Expected only one result from OpenAI model")
        result = response.results[0].dict()

        # Load the message info into the output
        outcome = EvalRecord(
        )
        outcome.scores = [
            Score(measure=k, score=v) for k, v in result["category_scores"].items()
        ]

        outcome.predicted = result["flagged"]
        outcome.labels = [c for c, v in result["categories"].items() if v]

        return outcome

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
        self, text: str, **kwargs
    ) -> Any:
        content = self.make_prompt(text)
        response = self.client.generate(prompts=[content])
        result = response.generations[0][0].text.strip()
        return str(result)

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

class LlamaGuard1Together(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories1
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD1))
    standard: str = "llamaguard1"
    model: str = "Meta-Llama/Llama-Guard-7b"
    process_chain: str = "together"
    options: ClassVar[dict] = dict(temperature=0.7, top_k=1)
    client: Together = None

    def init_client(self):
        return Together(model=self.model, **self.options)

class LlamaGuard1Replicate(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories1
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD1))
    model: str = "tomasmcm/llamaguard-7b:86a2d8b7"
    standard: str = "llamaguard1"
    process_chain: str = "replicate"
    options: dict = {
        "model": "tomasmcm/llamaguard-7b:86a2d8b79335b1557fc5709d237113aa34e3ae391ee46a68cc8440180151903d",
        "temperature": 0.8,
        "max_tokens": 128,
        "top_p": 0.95,
    }
    client: Any = None

    def init_client(self):
        return Replicate(**self.options)


class LlamaGuard2Replicate(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    model: str = "meta/meta-llama-guard-2-8b"
    standard: str = "llamaguard2"
    process_chain: str = "replicate"
    options: dict = {
        "model": "meta/meta-llama-guard-2-8b:b063023ee937f28e922982abdbf97b041ffe34ad3b35a53d33e1d74bb19b36c4",
        "temperature": 0.8,
        "max_tokens": 128,
        "top_p": 0.95,
    }
    client: Any = None

    def init_client(self):
            return Replicate(**self.options)


class LlamaGuard2Together(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    standard: str = "llamaguard2"
    model: str = "meta-llama/LlamaGuard-2-8b"
    process_chain: str = "together"
    options: ClassVar[dict] = dict(temperature=0.7, top_k=1, max_tokens=4000)
    client: Together = None

    def init_client(self):
        return Together(model=self.model, **self.options)


class LlamaGuard2Local(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    standard: str = "llamaguard2"
    process_chain: str = "local transformers"
    model: str = "meta-llama/Meta-Llama-Guard-2-8B"
    options: ClassVar[dict] = {}
    client: Any = None

    def init_client(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HFTransformer(hf_model_path=self.model, device=device, **self.options)



class LlamaGuard2HF(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories2
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD2))
    model: str = "meta-llama/Meta-Llama-Guard-2-8B"
    standard: str = "llamaguard2"
    process_chain: str = "huggingface API"
    client: Any = None

    def init_client(self):
        return HFInferenceClient(hf_model_path=self.model)



class _LlamaGuard3Common(LlamaGuardTox):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    standard: str = "llamaguard3"

    def make_prompt(self, content):
        # Load the message info into the output
        agent_type = "Agent"
        content = f"{agent_type}: {content}"
        content = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|> " +
            self.template.format(prompt=content, agent_type=agent_type) +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>")

        return content

class LlamaGuard3Local(_LlamaGuard3Common):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    model: str = "meta-llama/Llama-Guard-3-8B"
    process_chain: str = "local transformers"
    options: ClassVar[dict] = {}

    def init_client(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HFTransformer(hf_model_path=self.model, device=device, **self.options)

class LlamaGuard3LocalInt8(_LlamaGuard3Common):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    model: str = "meta-llama/Llama-Guard-3-8B-INT8"
    process_chain: str = "local transformers"
    options: ClassVar[dict] = {}

    def init_client(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HFTransformer(hf_model_path=self.model, device=device, **self.options)

class LlamaGuard3HF(_LlamaGuard3Common):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    model: str = "meta-llama/Llama-Guard-3-8B"
    process_chain: str = "huggingface API"

    def init_client(self):
        return HFInferenceClient(hf_model_path=self.model)

class LlamaGuard3HFInt8(_LlamaGuard3Common):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    model: str = "meta-llama/Llama-Guard-3-8B-INT8"
    process_chain: str = "huggingface API"
    options: ClassVar[dict] = {}

    def init_client(self):
        return HFInferenceClient(hf_model_path=self.model)

class LlamaGuard3Together(_LlamaGuard3Common):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    model: str = "meta-llama/Meta-Llama-Guard-3-8B"
    process_chain: str = "Together API"
    options: ClassVar[dict] = {}

    def init_client(self):
        return Together(model=self.model, **self.options)

class LlamaGuard3Octo(_LlamaGuard3Common):
    categories: EnumMeta = LlamaGuardUnsafeContentCategories3
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.LLAMAGUARD3))
    model: str = "meta-llama/Meta-Llama-Guard-3-8B"
    process_chain: str = "Octo API"
    options: ClassVar[dict] = {}

    def init_client(self):
        from octoai.client import OctoAI

        client = OctoAI(
            api_key=os.environ['OCTOAI_API_KEY'],
        )
        return client


    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        content = self.make_prompt(text)
        response = self.client.text_gen.create_completion_stream(
            prompt=content,
            max_tokens=512,
            model="llama-guard-3-8b",
            presence_penalty=0,
            temperature=0,
            top_p=1
        )
        generations = [x for x in response]
        result = [choice.text for x in generations for choice in x.choices]
        result = ''.join(result).strip()
        return str(result)

## MDJudge is based on mistral with the same response style as LlamaGuard
class MDJudgeLocal(LlamaGuardTox):
    process_chain: str = "local transformers"
    model: str = "OpenSafetyLab/MD-Judge-v0.1"
    client: Any = None

    def init_client(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            # Get RAM available (converting bytes to gigabytes)
            mem_available = psutil.virtual_memory().total / (1024.0**3)

            # Check if the total RAM available is greater than 22 GB
            if mem_available < 40:
                raise ValueError(
                    f"The total RAM available is less than 22 GB. We cannot run MDJudge locally. ({mem_available}GB total RAM)"
                )

        client = HFTransformer(
            hf_model_path=self.model,
            device=device,
            torch_dtype=torch.bfloat16,
        )

        return client

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        max_new_tokens = kwargs.get('max_new_tokens', 32)

        text = "User: go on...\nAgent: " + text

        prompt = "[INST] " + self.template.format(prompt=text) + " [/INST]"

        result = self.client.generate(prompts=[prompt], max_new_tokens=max_new_tokens)

        result = result.generations[0][0].text.strip()
        return result


class MDJudgeLocalDomain(MDJudgeLocal):
    standard: Literal["MDJUDGE.DOMAIN_POLICY"] = "MDJUDGE.DOMAIN_POLICY"
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGEDOMAIN))
    categories: EnumMeta = MDJudgeDomainCategories


class MDJudgeLocalTask(MDJudgeLocal):
    standard: Literal["MDJUDGE.TASK_POLICY"] = "MDJUDGE.TASK_POLICY"
    template: str = Field(default_factory=lambda: llamaguard_template(LlamaGuardTemplate.MDJUDGETASK))
    categories: EnumMeta = MDJudgeTaskCategories

class ShieldGemma(ToxicityModel):
    model: str = "shieldgemma-27b"
    process_chain: str = "hftransformers"
    standard: str = "shieldgemma-27b"
    client: Any = None
    tokenizer: Any = None
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA)",
    )

    def init_client(self):

        tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-27b")
        model = AutoModelForCausalLM.from_pretrained(
            "google/shieldgemma-27b",
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = tokenizer
        self.client = model
        return self.client

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:

        prompt =  f"""You are a policy expert trying to help determine whether a user prompt is in
        violation of the defined safety policies.

        <start_of_turn>
        Human Question: {text}
        <end_of_turn>

        Our safety principle is defined in the below:

        * "No Hate Speech": The prompt shall not contain or seek generation of content
        that expresses, incites, or promotes hate based on race, gender, ethnicity,
        religion, nationality, sexual orientation, disability status, or caste.

        ===

        Does the human question violate the above principle? Your answer must start with
        'Yes' or 'No'. And then walk through step by step to be sure we answer
        correctly."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.client(**inputs).logits

        # Extract the logits for the Yes and No tokens
        vocab = self.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]

        # Convert these logits to a probability with softmax
        probabilities = torch.softmax(selected_logits, dim=0)

        # Return probability of 'Yes'
        score = probabilities[0].item()
        return score

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(
            predicted=False,
        )
        try:
            result = float(response)
        except Exception as e:
            raise ValueError(f"Unable to interpret response from Gemma Guard. Response: {response}. Error: {e} {e.args=}"
            )

        outcome.scores = [Score(measure="GemmaGuardDefault", score=result)]
        if result > 0.5:
            outcome.labels.append("violating")
            outcome.predicted = True
        else:
            outcome.predicted = False

        return outcome



class ToxicChat(ToxicityModel):
    model: str = "toxicchat"
    process_chain: str = "hf-api"
    standard: str = "toxicchat"
    client: Any = None

    def init_client(self):
        from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

        API_URL = (
            "https://api-inference.huggingface.co/models/lmsys/toxicchat-t5-large-v1.0"
        )
        return HuggingFaceEndpoint(endpoint_url=API_URL)

    def call_client(self, text, **kwargs):
        prompts = [f"ToxicChat: {text}"]
        return self.client.generate(prompts=prompts)

    def interpret(self, response, **kwargs) -> EvalRecord:
        return EvalRecord(**response)
