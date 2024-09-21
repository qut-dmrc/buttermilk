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

from huggingface_hub import login
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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoModelForSequenceClassification

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
import datasets
from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.utils import read_text, read_yaml, scrub_serializable
from buttermilk.apis import HFInferenceClient, hf_pipeline
from buttermilk import logger
from ..types.tox import EvalRecord, Score
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
            self.init_client()
            if self.client is None:
                raise ValueError(f"Unable to initialize client for {self.model}")
        return self

    def init_client(self) -> None:
        if self.client is None:
            raise NotImplementedError()

    def run(self, job: Job) -> Job:
        response = self.moderate(content=job.record.content, record_id=job.record.record_id)
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

    @trace
    def __call__(self, prompt: str, **kwargs) -> EvalRecord:
        return self.moderate(prompt=prompt, **kwargs)

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
    def call_client(
        self, prompt: str, **kwargs
    ) -> EvalRecord:
        return self.client.__call__(prompt, **kwargs)

    @abc.abstractmethod
    def make_prompt(self, content):
        raise NotImplementedError

    @trace
    def moderate_batch(self, dataset, **kwargs):# -> Generator[Any, Any, None]:
        # if isinstance(self.client, transformers.Pipeline):
        #     # prepare batch
        #     dataset['text'] = dataset['text'].apply(self.make_prompt)
        #     input_ds = datasets.Dataset.from_pandas(dataset)
        #     for response in self.client(input_ds['text']):
        #         output = self.interpret(response)
        #         output = self.prepare_output_dict(output)
        #         yield output
        # el
        if isinstance(dataset, pd.DataFrame):
            for _, row in dataset.iterrows():
                # TODO: get this from the config instead
                record_id = row.get('id',row.get('record_id',row.get('name')))
                output = self.moderate(content=row['content'],record_id=record_id, **kwargs)
                output = output.model_dump()

                output = scrub_serializable(output)
                yield output
        else:
            for row in dataset:
                # TODO: get this from the config instead
                record_id = row.get('id',row.get('record_id',row.get('name')))
                output = self.moderate(content=row['content'],record_id=record_id, **kwargs)
                output = output.model_dump()

                output = scrub_serializable(output)
                yield output


    async def moderate_async(
            self, text: str, record_id:str, **kwargs
        ) -> EvalRecord:
            return self.moderate(content=text, record_id=record_id, **kwargs)

    @trace
    def moderate(
        self, content: str, record_id: str=None, **kwargs
    ) -> EvalRecord:
        prompt = self.make_prompt(content)

        response = self.call_client(prompt=prompt, **kwargs)

        try:
            output = self.interpret(response)
        except ValueError as e:
            err_msg = f"Unable to interpret response from {self.model}. Error: {e} {e.args=}"
            output = EvalRecord(
                error=err_msg, response=response)
            logger.error(err_msg)
        output = self.add_output_info(output, record_id=record_id)
        return output


    @trace
    @abc.abstractmethod
    def interpret(self, response: Any) -> EvalRecord:
        raise NotImplementedError()


    def add_output_info(self, record: EvalRecord, record_id=None, **kwargs) -> EvalRecord:
        # add identifying info in
        record.model=self.model
        record.process=self.process_chain
        record.standard=self.standard
        if record_id is not None:
            record.record_id=record_id

        return record

class _HF(ToxicityModel):
    process_chain: str = "local transformers"
    model: str
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA or auto)",
    )
    options: ClassVar[dict] = dict(temperature=1.0)
    call_options: ClassVar[dict] = dict(max_new_tokens=128)
    client: Any = None
    tokenizer: Any = None

    def init_client(self) -> None:
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Set a padding token

        self.client = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True).to(self.device)


    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:

        input_ids = self.tokenizer([prompt], padding="longest", return_tensors="pt").to(self.device)['input_ids']

        output = self.client.generate(input_ids=input_ids, **self.options, **self.call_options, **kwargs)
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

    # def batch(self, texts: list[str]):
    #     inputs = tokenizer(texts, padding="longest", padding_side="left", return_tensors="pt")
    #     inputs = {key: val.to(model.device) for key, val in inputs.items()}


class _Octo(ToxicityModel):
    model: str
    process_chain: str = "Octo API"
    options: ClassVar[dict]

    def init_client(self) -> None:
        from octoai.client import OctoAI

        client = OctoAI(
            api_key=os.environ['OCTOAI_API_KEY'],
        )
        self.client = client

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        response = self.client.text_gen.create_completion_stream(
            prompt=prompt,
            model=self.model,
            **self.options
        )
        generations = [x for x in response]
        result = [choice.text for x in generations for choice in x.choices]
        result = ''.join(result).strip()
        return str(result)



class Perspective(ToxicityModel):
    model: str = "perspective"
    process_chain: str = "api"
    standard: str = "perspective"
    client: Any = None

    def init_client(self) -> None:
        credentials, _ = google.auth.default()
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            credentials=credentials,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def make_prompt(self, content: str) -> str:
        return content

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
        self, prompt: str, **kwargs
    ) -> Any:
        if not (attributes := kwargs.get('attributes')):
            # get all
            attributes = (
                PerspectiveAttributes.__args__
                + PerspectiveAttributesExperimental.__args__
            )

        analyze_request = {
            "comment": {"text": prompt},
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

    def init_client(self) -> None:
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION")

        self.client = boto3.client(
            service_name="comprehend",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def make_prompt(self, content: str) -> str:
        return content

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        return self.client.detect_toxic_content(
            LanguageCode="en", TextSegments=[{"Text": prompt}]
        )

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(
            predicted=False,
        )
        result = response["ResultList"][0]

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

    def init_client(self) -> None:
        API_KEY = os.environ["AZURE_CONTENT_SAFETY_KEY"]
        ENDPOINT = os.environ.get(
            "AZURE_CONTENT_SAFETY_ENDPOINT",
            "https://westus.api.cognitive.microsoft.com",
        )

        credential = AzureKeyCredential(API_KEY)
        content_safety_client = ContentSafetyClient(ENDPOINT, credential)
        # blocklist_client = BlocklistClient(endpoint, credential)
        self.client = content_safety_client

    def make_prompt(self, content: str) -> str:
        return content

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        request = AnalyzeTextOptions(
            text=prompt, output_type=AnalyzeTextOutputType.EIGHT_SEVERITY_LEVELS
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


    def init_client(self) -> None:
        SUBSCRIPTION_KEY = os.environ["AZURE_CONTENT_MODERATOR_KEY"]
        ENDPOINT = os.environ.get(
            "AZURE_CONTENT_MODERATOR_ENDPOINT",
            "https://westus.api.cognitive.microsoft.com",
        )
        self.client = ContentModeratorClient(
            endpoint=ENDPOINT,
            credentials=CognitiveServicesCredentials(subscription_key=SUBSCRIPTION_KEY),
        )

    def make_prompt(self, content: str) -> str:
        return content

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        text_stream = StringIO(prompt)
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

    def init_client(self) -> None:
        if torch.cuda.is_available():
            self.client = evaluate.load("regard", module_type="measurement",device = "cuda")
        else:
            self.client = evaluate.load("regard", module_type="measurement")


    def make_prompt(self, content: str) -> str:
        return content

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        result = self.client.compute(data=[prompt])
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

    def init_client(self) -> None:
        self.client = evaluate.load("honest", "en")

    def make_prompt(self, content: str) -> list[str]:
        return content.split(' ')

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        result = self.client.compute(predictions=prompt)
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
    model: str = "facebook/roberta-hate-speech-dynabench-r4-target"
    process_chain: str = "hf_transformers"
    standard: str = "lftw_r4_target"
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA or auto)",
    )
    options: ClassVar[dict] = dict()
    tokenizer: Any = None
    classes: dict = {}

    def init_client(self) -> None:
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Set a padding token
        cfg = AutoConfig.from_pretrained(self.model)
        self.classes = cfg.id2label
        self.client = AutoModelForSequenceClassification.from_pretrained(self.model).to(self.device)


    def make_prompt(self, content: str) -> str:
        return content

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:

        input_ids = self.tokenizer([prompt], return_tensors="pt").to(self.device)['input_ids']
        with torch.no_grad():
            response = self.client(input_ids=input_ids, **self.options, **kwargs)
        logits = response.logits
        predicted_class_id = logits.argmax().item()
        result = self.classes[predicted_class_id]
        confidence = float(logits.softmax(dim=-1)[0][predicted_class_id])
        return dict(label=result, confidence=confidence)

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        # Load the message info into the output
        outcome = EvalRecord(
        )
        outcome.scores = [Score(measure=response['label'], confidence=response["confidence"])]
        outcome.predicted = response['label'] == "hate"
        outcome.labels = [response['label'] ]

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
        "need caution": 2,
        "needs intervention": 3,
        "possibly needs caution": 4,
        "probably needs caution": 5,
    }

    def init_client(self) -> None:
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.client = hf_pipeline(hf_model_path="togethercomputer/GPT-JT-Moderation-6B",
                device=self.device, max_new_tokens=3
        )

    def make_prompt(self, content: str) -> str:
        prompt = self.template.format(content=content)
        return prompt

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        response = self.client(prompt)

        if len(response) > 1:
            raise ValueError("Expected only one result from model")
        result = response[0]['generated_text']
        if result.startswith(prompt):
            result = result[len(prompt):]

        return result.strip()

    @trace
    def interpret(self, response: Any) -> EvalRecord:

        # Load the message info into the output
        outcome = EvalRecord(
        )
        try:
            outcome.response = str(response)
            outcome.scores = [
                Score(
                    measure=self.standard, score=self.ResponseMap[response], labels=[response]
                )
            ]

            outcome.predicted = self.ResponseMap[response] >= 2
            outcome.labels = [response]
        except Exception as e:
            raise ValueError(f"Unable to interpret response from GPT-JT model. {response=}, {e=}, {e.args=}")

        return outcome


###
# todo: add https://github.com/unitaryai/detoxify
####


class OpenAIModerator(ToxicityModel):
    model: str = "text-moderation-latest"
    process_chain: str = "api"
    standard: str = "openaimod"
    client: Any = None

    def init_client(self) -> None:
        openai.api_type = 'openai'
        self.client = openai.moderations

    def make_prompt(self, content: str) -> str:
        return content

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:
        return self.client.create(input=prompt, model=self.model)

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

class ShieldGemma(ToxicityModel):
    model: str = "google/shieldgemma-27b"
    process_chain: str = "local transformers"
    standard: str = "shieldgemma"
    client: transformers.Pipeline = None
    tokenizer: Any = None
    classes: Any = None
    template: str = Field(default_factory=lambda: read_text(TEMPLATE_DIR / "shieldgemma.txt"))
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA or auto)",
    )

    def init_client(self) -> None:
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.client = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype=torch.bfloat16)
        vocab = self.tokenizer.get_vocab()
        self.classes = [vocab['Yes'], vocab['No']]


    def make_prompt(self, text):
        prompt =  self.template.format(text=text)
        return prompt

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            model_outputs = self.client(**inputs)

        logits = model_outputs.logits

        selected_logits = logits[0, -1, self.classes]

        # Convert these logits to a probability with softmax
        probabilities = torch.softmax(selected_logits, dim=0)

        # Return probability of 'Yes'
        score = float(probabilities[0].item())

        return dict(score=score)

    @trace
    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(
            predicted=False,
        )

        score = response['score']
        outcome.scores = [Score(measure="GemmaGuardDefault", score=score)]
        if score > 0.5:
            outcome.labels.append("violating")
            outcome.predicted = True
        else:
            outcome.predicted = False

        return outcome


class ShieldGemma2b(ShieldGemma):
    model: str = "google/shieldgemma-2b"

class ShieldGemma9b(ShieldGemma):
    model: str = "google/shieldgemma-9b"

class ToxicChat(ToxicityModel):
    model: str = "toxicchat"
    process_chain: str = "hf-api"
    standard: str = "toxicchat"
    client: Any = None

    def init_client(self) -> None:
        from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

        API_URL = (
            "https://api-inference.huggingface.co/models/lmsys/toxicchat-t5-large-v1.0"
        )
        self.client = HuggingFaceEndpoint(endpoint_url=API_URL)

    def interpret(self, response, **kwargs) -> EvalRecord:
        return EvalRecord(**response)
